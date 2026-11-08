package com.thirdeye.capture

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.wifi.WifiManager
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.text.InputType
import android.util.Log
import android.util.Range
import android.util.Size
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.View
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.HttpURLConnection
import java.net.InetSocketAddress
import java.net.Socket
import java.net.URL
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * Second camera for the Jenga tracker, the Pixel counterpart of lidar-capture/ViewController.swift.
 *
 * No LiDAR here: the Pixel streams plain RGB frames to the capture computer's POST /android (the same endpoint
 * the browser /cam page uses), and the computer saves them while the iPhone is recording.
 *
 * Streaming is pipelined for Wi-Fi: the camera thread only copies each frame into a pooled bitmap, and up to
 * [uploadsInFlight] workers encode and send frames in parallel, each over its own kept-alive socket. A frame that
 * arrives while the pipeline is full is dropped. Frames carry a sequence number so the server can drop one that
 * overtook a newer frame.
 * START / STOP / MARK go through the capture computer's /cmd/ endpoints, which relay them to the iPhone,
 * so either phone (or the browser viewer) can drive the recording.
 *
 * Wired mode (the counterpart of the iPhone's UsbRelay.swift, but simpler: Android needs no relay). With a USB cable
 * and `adb reverse tcp:8000 tcp:8000` (scripts/wired.sh) the game server shows up as 127.0.0.1:8000 on the phone, so
 * no network is involved at all. Once a second the app checks whether the server answers there; while it does,
 * everything goes over the cable and the IP box turns green. When it stops answering for 3 s, back to Wi-Fi.
 */
class MainActivity : ComponentActivity() {
    private val tag = "JengaCam"
    private val maxFps = 60               // the Pixel's back camera does a fixed 60; falls back to 30 if not
    private val uploadsInFlight = 4       // JPEG encode alone is 10-19 ms, so at 60 fps it has to run in parallel
    private val jpegQuality = 70          // ~40% smaller than q=80, still fine for ArUco at 1280×720
    private val frameSize = Size(1280, 720)  // what camera_android.json was calibrated at
    private val wiredHost = "127.0.0.1"   // the computer's port 8000 after `adb reverse tcp:8000 tcp:8000`

    private lateinit var previewView: PreviewView
    private lateinit var ipAddressField: EditText
    private lateinit var statusLabel: TextView
    private lateinit var startStopButton: Button
    private lateinit var markButton: Button

    private lateinit var frameExecutor: ExecutorService
    private val cmdExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private val wireProbe: ScheduledExecutorService = Executors.newSingleThreadScheduledExecutor()
    private val uploadExecutor: ExecutorService = Executors.newFixedThreadPool(uploadsInFlight)
    private val inFlight = AtomicInteger(0)
    private val links = ThreadLocal.withInitial { FrameLink() }   // one kept-alive socket per upload thread
    private val jpegOut = ThreadLocal.withInitial { ByteArrayOutputStream(256 * 1024) }
    private var wifiLock: WifiManager.WifiLock? = null

    // Reused frame bitmaps, so 60 fps doesn't churn 3.6 MB allocations through the GC; at most one per worker is out.
    private val freeBitmaps = ArrayBlockingQueue<Bitmap>(uploadsInFlight + 1)

    // Frame thread only: what the camera actually delivers, to tell a slow sensor from a slow link.
    private var frameSeq = 0L
    private var camFrames = 0
    private var camSince = 0L
    @Volatile private var cameraFps = 0

    // Written on the frame thread, read on the main thread.
    @Volatile private var macIP = ""
    @Volatile private var wired = false   // the game server answers on 127.0.0.1 over the USB cable
    @Volatile private var lastWiredOk = 0L
    @Volatile private var lastUploadMs = 0L
    private var sentSize = ""
    private var isRecording = false
    private var markers: List<Int> = emptyList()
    private var saved = 0
    private var reachable = false
    private var framesSent = 0
    private val sentTimes = ArrayDeque<Long>()   // main thread: reply times over the last ~2 s, for the fps readout
    private var uploadsFailed = 0
    private var wantSince = 0L            // when START/STOP was tapped, until the computer confirms it
    private var want: Boolean? = null
    private var markCount = 0

    // Android 17 (API 37) blocks connections to the local network until this is granted; older versions don't have it.
    private val localNetworkPermission = "android.permission.ACCESS_LOCAL_NETWORK"
    private val neededPermissions =
        if (Build.VERSION.SDK_INT >= 37) arrayOf(Manifest.permission.CAMERA, localNetworkPermission)
        else arrayOf(Manifest.permission.CAMERA)

    private val requestPermissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { res ->
        val camera = res[Manifest.permission.CAMERA] ?: isGranted(Manifest.permission.CAMERA)
        // A denied network permission shows up later as "Cannot reach the computer" (see refreshUi).
        if (camera) startCamera() else setStatus("❌ Camera permission denied\nAllow it in Settings → Apps", Color.RED)
    }

    private fun isGranted(p: String) = ContextCompat.checkSelfPermission(this, p) == PackageManager.PERMISSION_GRANTED

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        frameExecutor = Executors.newSingleThreadExecutor()
        // Without this the Pixel's Wi-Fi power save batches packets, adding tens of ms per frame.
        wifiLock = applicationContext.getSystemService(WifiManager::class.java)
            ?.createWifiLock(WifiManager.WIFI_MODE_FULL_LOW_LATENCY, "JengaCam:stream")
            ?.apply { setReferenceCounted(false) }
        buildUi()
        wireProbe.scheduleWithFixedDelay(::probeWire, 0, 1, TimeUnit.SECONDS)

        if (neededPermissions.all(::isGranted)) {
            startCamera()
        } else {
            requestPermissions.launch(neededPermissions)
        }
    }

    override fun onResume() {
        super.onResume()
        wifiLock?.acquire()
    }

    override fun onPause() {
        super.onPause()
        wifiLock?.release()
    }

    override fun onDestroy() {
        super.onDestroy()
        frameExecutor.shutdown()
        cmdExecutor.shutdown()
        wireProbe.shutdownNow()
        uploadExecutor.shutdown()
    }

    // MARK: - UI (built in code, like the iPhone app)
    private fun buildUi() {
        val dp = resources.displayMetrics.density
        val root = FrameLayout(this).apply { setBackgroundColor(Color.BLACK) }

        previewView = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            setOnClickListener { dismissKeyboard() }
        }
        root.addView(previewView, FrameLayout.LayoutParams(-1, -1))

        val prefs = getSharedPreferences("settings", Context.MODE_PRIVATE)
        ipAddressField = EditText(this).apply {
            hint = "Computer IP (printed by the server)"
            setText(prefs.getString("macIP", "192.168.0.54"))
            setTextColor(Color.BLACK)
            setHintTextColor(Color.GRAY)
            gravity = Gravity.CENTER
            isSingleLine = true
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            imeOptions = EditorInfo.IME_ACTION_DONE
            background = rounded(Color.argb(230, 255, 255, 255), 8 * dp)
            setOnEditorActionListener { _, action, _ ->
                if (action == EditorInfo.IME_ACTION_DONE) { dismissKeyboard(); true } else false
            }
            setOnFocusChangeListener { _, focused -> if (!focused) saveIP() }
        }
        macIP = ipAddressField.text.toString().trim()

        statusLabel = TextView(this).apply {
            text = "Starting camera…"
            setTextColor(Color.WHITE)
            textSize = 15f
            typeface = Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setPadding((8 * dp).toInt(), (6 * dp).toInt(), (8 * dp).toInt(), (6 * dp).toInt())
            background = rounded(Color.argb(180, 0, 0, 0), 8 * dp)
        }

        val top = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(ipAddressField, LinearLayout.LayoutParams(-1, (44 * dp).toInt()))
            addView(statusLabel, LinearLayout.LayoutParams(-1, -2).apply { topMargin = (8 * dp).toInt() })
        }
        root.addView(top, FrameLayout.LayoutParams((320 * dp).toInt(), -2, Gravity.TOP or Gravity.START).apply {
            setMargins((20 * dp).toInt(), (20 * dp).toInt(), 0, 0)
        })

        markButton = bigButton("MARK MOVE DONE", Color.rgb(255, 149, 0)) { markMove() }
        startStopButton = bigButton("START CAPTURE", Color.rgb(52, 199, 89)) { toggleCapture() }
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(markButton, LinearLayout.LayoutParams((200 * dp).toInt(), (60 * dp).toInt()))
            addView(startStopButton, LinearLayout.LayoutParams((200 * dp).toInt(), (60 * dp).toInt()).apply {
                topMargin = (20 * dp).toInt()
            })
        }
        root.addView(buttons, FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM or Gravity.END).apply {
            setMargins(0, 0, (24 * dp).toInt(), (24 * dp).toInt())
        })

        setContentView(root)
    }

    private fun bigButton(title: String, color: Int, onTap: () -> Unit) = Button(this).apply {
        text = title
        textSize = 17f
        typeface = Typeface.DEFAULT_BOLD
        setTextColor(Color.WHITE)
        isAllCaps = false
        stateListAnimator = null
        background = rounded(color, 12 * resources.displayMetrics.density)
        setOnClickListener { onTap() }
    }

    private fun rounded(color: Int, radius: Float) = GradientDrawable().apply {
        setColor(color)
        cornerRadius = radius
    }

    private fun setStatus(text: String, bg: Int) {
        statusLabel.text = text
        (statusLabel.background as GradientDrawable).setColor(bg)
    }

    private fun dismissKeyboard() {
        ipAddressField.clearFocus()
        getSystemService(InputMethodManager::class.java)
            .hideSoftInputFromWindow(ipAddressField.windowToken, 0)
    }

    /** Where frames and commands go: the cable when it works, otherwise the typed IP over Wi-Fi. */
    private fun host() = if (wired) wiredHost else macIP

    // On wireProbe, once a second: is the game server there over the cable? A plain TCP connect is not enough,
    // because adb accepts the connection on the phone even when nothing listens on the computer.
    private fun probeWire() {
        val ok = try {
            val c = URL("http://$wiredHost:8000/status.json").openConnection() as HttpURLConnection
            c.connectTimeout = 500
            c.readTimeout = 1500
            val code = c.responseCode
            c.inputStream.use { it.readBytes() }
            code == 200
        } catch (e: Exception) {
            false
        }
        val now = SystemClock.elapsedRealtime()
        if (ok) lastWiredOk = now
        val w = lastWiredOk != 0L && now - lastWiredOk < 3000
        if (w != wired) {
            wired = w
            Log.i(tag, if (w) "USB cable in use" else "USB cable gone, back to Wi-Fi")
            runOnUiThread { showWired(); refreshUi() }
        }
    }

    private fun showWired() {
        (ipAddressField.background as GradientDrawable).setColor(
            if (wired) Color.argb(230, 52, 199, 89) else Color.argb(230, 255, 255, 255))
        ipAddressField.hint = if (wired) "USB cable connected" else "Computer IP (printed by the server)"
    }

    private fun saveIP() {
        val ip = ipAddressField.text.toString().trim()
        macIP = ip
        if (ip.isNotEmpty()) {
            getSharedPreferences("settings", Context.MODE_PRIVATE).edit().putString("macIP", ip).apply()
        }
    }

    // MARK: - Controls (relayed to the iPhone by the computer)
    private fun toggleCapture() {
        dismissKeyboard()
        if (host().isEmpty()) {
            setStatus("Enter Computer IP (or plug in the USB cable) first!", Color.argb(200, 255, 59, 48))
            return
        }
        val start = !(want ?: isRecording)
        want = start
        wantSince = SystemClock.elapsedRealtime()
        startStopButton.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
        if (start) markCount = 0
        sendCommand(if (start) "start" else "stop")
        refreshUi()
    }

    private fun markMove() {
        if (!isRecording) {
            setStatus("Press START first,\nthen MARK after each move", Color.argb(180, 0, 0, 0))
            return
        }
        markCount += 1
        markButton.performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
        sendCommand("mark")
        Log.i(tag, "MARK #$markCount")
    }

    private fun sendCommand(cmd: String) {
        val ip = host()
        cmdExecutor.execute {
            try {
                val c = URL("http://$ip:8000/cmd/$cmd").openConnection() as HttpURLConnection
                c.requestMethod = "POST"
                c.connectTimeout = 2000
                c.readTimeout = 2000
                c.doOutput = true
                c.setFixedLengthStreamingMode(0)
                c.outputStream.close()
                val code = c.responseCode
                c.inputStream.close()
                Log.i(tag, "cmd $cmd -> HTTP $code")
            } catch (e: Exception) {
                Log.w(tag, "cmd $cmd failed: $e")
                runOnUiThread {
                    want = null
                    setStatus("❌ Cannot reach the computer at $ip\n($cmd not sent)", Color.argb(200, 255, 59, 48))
                }
            }
        }
    }

    // MARK: - Camera
    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(frameSize, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            fun bind(fps: Int?) {
                val preview = Preview.Builder().setResolutionSelector(selector)
                    // A fixed range: indoors auto-exposure may otherwise drop the sensor to 15 fps.
                    .apply { if (fps != null) setTargetFrameRate(Range(fps, fps)) }
                    .build().also { it.surfaceProvider = previewView.surfaceProvider }
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(selector)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(frameExecutor) { image -> onFrame(image) }
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }
            try {
                val fps = listOf(maxFps, 30).firstOrNull { f ->
                    try { bind(f); true } catch (e: Exception) { Log.w(tag, "$f fps not supported", e); false }
                }
                if (fps == null) bind(null)
                Log.i(tag, "camera bound at ${fps ?: "default"} fps")
                setStatus("Camera ready\nConnecting to the computer…", Color.argb(180, 0, 0, 0))
            } catch (e: Exception) {
                Log.e(tag, "camera bind failed", e)
                setStatus("❌ Camera failed: ${e.message}", Color.argb(200, 255, 59, 48))
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // Runs on frameExecutor: only copy the frame here; encoding and upload run on uploadExecutor, several at once.
    private fun onFrame(image: ImageProxy) {
        val now = SystemClock.elapsedRealtime()
        camFrames += 1
        if (now - camSince >= 2000) {
            cameraFps = (camFrames * 1000 / (now - camSince).coerceAtLeast(1)).toInt()
            camFrames = 0
            camSince = now
        }
        val ip = host()
        // No minimum gap between sends: the camera's fixed frame rate is the cap, and CameraX delivers frames
        // unevenly, so a gap check dropped one frame in five at 60 fps.
        if (ip.isEmpty() || inFlight.get() >= uploadsInFlight) {
            image.close()
            return
        }
        val bmp: Bitmap
        val rot: Int
        try {
            bmp = toBitmap(image)
            rot = image.imageInfo.rotationDegrees
        } finally {
            image.close()
        }
        val drawMs = SystemClock.elapsedRealtime() - now
        val seq = ++frameSeq
        inFlight.incrementAndGet()
        uploadExecutor.execute {
            try {
                val t0 = SystemClock.elapsedRealtime()
                val frame = if (rot == 0) bmp
                    else Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(rot.toFloat()) }, true)
                val out = jpegOut.get()!!
                out.reset()
                frame.compress(Bitmap.CompressFormat.JPEG, jpegQuality, out)
                val jpeg = out.toByteArray()
                val encMs = SystemClock.elapsedRealtime() - t0
                freeBitmaps.offer(bmp)
                upload(ip, jpeg, "${frame.width}×${frame.height}", seq, drawMs, encMs)
            } finally {
                inFlight.decrementAndGet()
            }
        }
    }

    /** RGBA plane copied into a pooled bitmap; falls back to CameraX's converter for padded rows. */
    private fun toBitmap(image: ImageProxy): Bitmap {
        val plane = image.planes[0]
        if (plane.pixelStride != 4 || plane.rowStride != image.width * 4) return image.toBitmap()
        val bmp = freeBitmaps.poll()?.takeIf { it.width == image.width && it.height == image.height }
            ?: Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
        plane.buffer.rewind()
        bmp.copyPixelsFromBuffer(plane.buffer)
        return bmp
    }

    private fun upload(ip: String, jpeg: ByteArray, size: String, seq: Long, drawMs: Long, encMs: Long) {
        val t0 = SystemClock.elapsedRealtime()
        try {
            val body = links.get()!!.post(ip, 8000, "/android", jpeg, mapOf(
                "Content-Type" to "image/jpeg",
                "X-Frame" to seq.toString(),
                "X-Draw" to drawMs.toString(),
                "X-Enc" to encMs.toString(),
                "X-Up" to lastUploadMs.toString()))
            lastUploadMs = SystemClock.elapsedRealtime() - t0
            val j = JSONObject(body)
            val ids = j.optJSONArray("markers")
            val m = if (ids == null) emptyList() else List(ids.length()) { ids.getInt(it) }
            val rec = j.optBoolean("recording")
            val s = j.optInt("saved")
            val kb = jpeg.size / 1024
            runOnUiThread { sentSize = size; onReply(m, rec, s, kb) }
        } catch (e: Exception) {
            Log.w(tag, "upload failed: $e")
            runOnUiThread {
                uploadsFailed += 1
                reachable = false
                refreshUi()
            }
        }
    }

    /**
     * Minimal HTTP/1.1 client on one kept-alive socket with TCP_NODELAY: request line, headers and body go out
     * in a single write, so there is no per-frame handshake and no Nagle stall (HttpURLConnection allows neither).
     */
    private class FrameLink {
        private var socket: Socket? = null
        private var input: InputStream? = null
        private var host = ""

        fun post(ip: String, port: Int, path: String, body: ByteArray, headers: Map<String, String>): String {
            // A kept-alive socket the server already closed fails on first use; retry once on a fresh one.
            val reused = socket != null && host == ip
            return try {
                exchange(ip, port, path, body, headers)
            } catch (e: Exception) {
                close()
                if (!reused) throw e
                exchange(ip, port, path, body, headers)
            }
        }

        private fun exchange(ip: String, port: Int, path: String, body: ByteArray, headers: Map<String, String>): String {
            if (socket == null || host != ip) {
                close()
                val s = Socket()
                s.tcpNoDelay = true
                s.sendBufferSize = 512 * 1024
                s.connect(InetSocketAddress(ip, port), 2000)
                s.soTimeout = 3000
                socket = s
                input = BufferedInputStream(s.getInputStream(), 16 * 1024)
                host = ip
            }
            val head = StringBuilder("POST $path HTTP/1.1\r\nHost: $ip:$port\r\nConnection: keep-alive\r\n")
            headers.forEach { (k, v) -> head.append(k).append(": ").append(v).append("\r\n") }
            head.append("Content-Length: ").append(body.size).append("\r\n\r\n")
            val h = head.toString().toByteArray()
            val req = ByteArray(h.size + body.size)
            System.arraycopy(h, 0, req, 0, h.size)
            System.arraycopy(body, 0, req, h.size, body.size)
            socket!!.getOutputStream().apply { write(req); flush() }

            val inp = input!!
            val status = readLine(inp)
            val code = status.split(" ").getOrNull(1)?.toIntOrNull() ?: throw IllegalStateException("bad reply: $status")
            var length = 0
            var closeAfter = false
            while (true) {
                val line = readLine(inp)
                if (line.isEmpty()) break
                val i = line.indexOf(':')
                if (i < 0) continue
                val k = line.substring(0, i).trim().lowercase()
                val v = line.substring(i + 1).trim()
                if (k == "content-length") length = v.toInt()
                if (k == "connection" && v.equals("close", ignoreCase = true)) closeAfter = true
            }
            val buf = ByteArray(length)
            var n = 0
            while (n < length) {
                val r = inp.read(buf, n, length - n)
                if (r < 0) throw IllegalStateException("connection closed mid-reply")
                n += r
            }
            if (closeAfter) close()
            val text = buf.decodeToString()
            if (code != 200) throw IllegalStateException("HTTP $code $text")
            return text
        }

        private fun readLine(inp: InputStream): String {
            val sb = StringBuilder()
            while (true) {
                val c = inp.read()
                if (c < 0) throw IllegalStateException("connection closed")
                if (c == '\n'.code) break
                if (c != '\r'.code) sb.append(c.toChar())
            }
            return sb.toString()
        }

        fun close() {
            try { socket?.close() } catch (_: Exception) {}
            socket = null
            input = null
        }
    }

    // MARK: - State from the computer
    private fun onReply(m: List<Int>, rec: Boolean, s: Int, kb: Int) {
        reachable = true
        framesSent += 1
        val now = SystemClock.elapsedRealtime()
        sentTimes.addLast(now)
        while (sentTimes.first() < now - 2000) sentTimes.removeFirst()
        markers = m
        saved = s
        if (rec != isRecording) {
            isRecording = rec
            startStopButton.performHapticFeedback(HapticFeedbackConstants.CONFIRM)
        }
        if (want != null && want == rec) want = null
        refreshUi(kb)
    }

    private fun fps(): String {
        val t = sentTimes
        return if (t.size < 2) "–" else "%.0f".format((t.size - 1) * 1000.0 / (t.last() - t.first()).coerceAtLeast(1))
    }

    private fun refreshUi(kb: Int = 0) {
        val pending = want
        val waitedMs = SystemClock.elapsedRealtime() - wantSince
        val showStop = pending ?: isRecording
        startStopButton.text = if (showStop) "STOP CAPTURE" else "START CAPTURE"
        (startStopButton.background as GradientDrawable).setColor(
            if (showStop) Color.rgb(255, 59, 48) else Color.rgb(52, 199, 89))

        if (!reachable) {
            val hint = if (!neededPermissions.all(::isGranted)) "Allow Nearby devices in Settings → Apps → Permissions"
                else "Is iphone_saved.py running? Same Wi-Fi?\nOr plug in the USB cable and run scripts/wired.sh"
            setStatus("❌ Cannot reach the computer at ${host()}:8000\n$hint", Color.argb(200, 255, 59, 48))
            return
        }
        val link = if (wired) "🔌 USB" else "Wi-Fi"
        val markerText = "markers: " + (markers.joinToString(", ").ifEmpty { "none" })
        when {
            pending != null && waitedMs > 4000 -> setStatus(
                "⚠️ Waiting for the iPhone…\nIs the LiDAR app open on the iPhone?\n$markerText",
                Color.argb(200, 255, 149, 0))
            pending != null -> setStatus(
                (if (pending) "Starting iPhone recording…" else "Stopping…") + "\n$markerText",
                Color.argb(180, 0, 0, 0))
            isRecording -> setStatus(
                "🔴 RECORDING · $link · ${fps()} fps (camera $cameraFps)\nSaved: $saved  ·  marks: $markCount\n$markerText",
                Color.argb(200, 255, 59, 48))
            else -> setStatus(
                "✅ Connected · $link · $sentSize · $kb KB · ${fps()} fps (camera $cameraFps)\n$markerText\nTap START (or START on the computer)",
                if (markers.size >= 2) Color.argb(180, 52, 199, 89) else Color.argb(200, 255, 149, 0))
        }
    }
}
