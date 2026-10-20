//
//  ViewController.swift
//  lidar-capture
//
//  Created by Mitul on 9/27/26.
//

import UIKit
import ARKit
import SceneKit
import AVFoundation

class ViewController: UIViewController, ARSCNViewDelegate, ARSessionDelegate {
    var sceneView: ARSCNView!
    var isCapturing = false
    var frameCount = 0
    var totalFrames = 0
    var rejectedTracking = 0
    let saveEvery = 20  // ~3 saved frames/s at 60 fps; plenty for a game, ~half the data of every-10th
    let previewQueue = DispatchQueue(label: "com.lidarcapture.preview")
    let previewContext = CIContext()
    var previewBusy = false
    var uploadsAttempted = 0
    var uploadsFailed = 0
    var statusLabel: UILabel!
    var startStopButton: UIButton!
    var markButton: UIButton!
    var markCount = 0
    var lastMarkSeq: Int? = nil
    var meshTest = false
    var lastMeshExport = Date.distantPast
    var ipAddressField: UITextField!
    let saveQueue = DispatchQueue(label: "com.lidarcapture.savequeue")
    let sessionID = UUID().uuidString  // new value each app launch — frames from different launches don't share a world origin

    var macServerURL = "http://10.16.40.91:8000/upload"

    override func viewDidLoad() {
        super.viewDidLoad()
        
        view.backgroundColor = .black
        print("✅ ViewController loaded")
        print("🆕🆕🆕 BUILD MARKER: POSE-V2 🆕🆕🆕")

        // ARSCNView
        sceneView = ARSCNView(frame: view.bounds)
        sceneView.delegate = self
        sceneView.session.delegate = self  // CRITICAL: session delegate
        sceneView.autoenablesDefaultLighting = true
        sceneView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        sceneView.showsStatistics = true
        view.addSubview(sceneView)
        
        // IP field
        ipAddressField = UITextField(frame: CGRect(x: 20, y: 50, width: view.bounds.width - 40, height: 40))
        ipAddressField.placeholder = "Mac IP (printed by the server)"
        ipAddressField.text = UserDefaults.standard.string(forKey: "macIP") ?? "192.168.0.54"
        ipAddressField.backgroundColor = UIColor.white.withAlphaComponent(0.9)
        ipAddressField.textAlignment = .center
        ipAddressField.keyboardType = .numbersAndPunctuation
        ipAddressField.layer.cornerRadius = 8
        ipAddressField.autocorrectionType = .no
        ipAddressField.autocapitalizationType = .none
        ipAddressField.returnKeyType = .done
        ipAddressField.delegate = self
        view.addSubview(ipAddressField)
        
        // Status label
        statusLabel = UILabel(frame: CGRect(x: 20, y: 100, width: view.bounds.width - 40, height: 100))
        statusLabel.numberOfLines = 4
        statusLabel.text = "Initializing AR...\nWait for depth data\nThen tap START"
        statusLabel.textColor = .white
        statusLabel.font = UIFont.systemFont(ofSize: 16, weight: .bold)
        statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        statusLabel.layer.cornerRadius = 8
        statusLabel.clipsToBounds = true
        statusLabel.textAlignment = .center
        view.addSubview(statusLabel)
        
        // Button
        startStopButton = UIButton(type: .system)
        startStopButton.frame = CGRect(x: (view.bounds.width - 200) / 2,
                                       y: view.bounds.height - 120,
                                       width: 200,
                                       height: 60)
        startStopButton.setTitle("START CAPTURE", for: .normal)
        startStopButton.titleLabel?.font = UIFont.systemFont(ofSize: 18, weight: .bold)
        startStopButton.backgroundColor = UIColor.systemGreen
        startStopButton.setTitleColor(.white, for: .normal)
        startStopButton.layer.cornerRadius = 12
        startStopButton.addTarget(self, action: #selector(toggleCapture), for: .touchUpInside)
        view.addSubview(startStopButton)

        markButton = UIButton(type: .system)
        markButton.frame = CGRect(x: (view.bounds.width - 200) / 2, y: view.bounds.height - 200, width: 200, height: 60)
        markButton.setTitle("MARK MOVE DONE", for: .normal)
        markButton.titleLabel?.font = UIFont.systemFont(ofSize: 17, weight: .bold)
        markButton.backgroundColor = UIColor.systemOrange
        markButton.setTitleColor(.white, for: .normal)
        markButton.layer.cornerRadius = 12
        markButton.addTarget(self, action: #selector(markMove), for: .touchUpInside)
        view.addSubview(markButton)
        
        print("✅ UI created")

        // Check support
        guard ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) else {
            print("❌ No mesh reconstruction support")
            showAlert(message: "This device doesn't support LiDAR mesh reconstruction")
            return
        }
        
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            print("❌ No scene depth support")
            showAlert(message: "This device doesn't support scene depth")
            return
        }
        
        print("✅ LiDAR and scene depth supported")

        // Configure AR
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]  // Use array syntax
        
        print("✅ Starting AR session...")
        sceneView.session.run(config)

        print("✅ Setup complete!")

        // Watchdog: if tracking hasn't normalized in 8s, the camera is probably
        // being held by something else (Continuity Camera, another app, etc.)
        DispatchQueue.main.asyncAfter(deadline: .now() + 8) { [weak self] in
            guard let self = self, let frame = self.sceneView.session.currentFrame else { return }
            if case .limited = frame.camera.trackingState {
                self.showAlert(message: """
                    Camera isn't initializing after 8s.

                    Likely cause: another app or Continuity Camera is using \
                    the camera. Try:
                    • Quit Zoom/FaceTime/QuickTime/etc. on your Mac
                    • Settings → General → AirPlay & Handoff → turn off \
                    Continuity Camera
                    • Restart the phone if that doesn't help
                    """)
            }
        }
        
        // Keyboard dismissal
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(dismissKeyboard))
        tapGesture.cancelsTouchesInView = false
        view.addGestureRecognizer(tapGesture)
    }
    
    @objc func dismissKeyboard() {
        view.endEditing(true)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        startStopButton.frame = CGRect(x: (view.bounds.width - 200) / 2,
                                       y: view.bounds.height - 120,
                                       width: 200,
                                       height: 60)
        markButton.frame = CGRect(x: (view.bounds.width - 200) / 2, y: view.bounds.height - 200, width: 200, height: 60)
        statusLabel.frame = CGRect(x: 20, y: 100, width: view.bounds.width - 40, height: 100)
        ipAddressField.frame = CGRect(x: 20, y: 50, width: view.bounds.width - 40, height: 40)
    }
    
    func showAlert(message: String) {
        let alert = UIAlertController(title: "Error", message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }

    @objc func toggleCapture() {
        view.endEditing(true)
        
        if let ip = ipAddressField.text, !ip.isEmpty {
            macServerURL = "http://\(ip):8000/upload"
        } else {
            showAlert(message: "Enter Mac IP first!")
            return
        }
        
        isCapturing.toggle()
        AudioServicesPlaySystemSound(isCapturing ? 1519 : 1520)
        
        print("========================================")
        if isCapturing {
            print("🔴 CAPTURE STARTED - \(macServerURL)")
            rejectedTracking = 0; uploadsAttempted = 0; uploadsFailed = 0; markCount = 0
        } else {
            print("⏸️ STOPPED")
            print("   Frames sent for upload : \(frameCount/saveEvery)")
            print("   Dropped (bad tracking) : \(rejectedTracking)")
            print("   Upload requests        : \(uploadsAttempted)")
            print("   Upload failures        : \(uploadsFailed)")
        }
        print("========================================")
        
        if isCapturing {
            startStopButton.setTitle("STOP CAPTURE", for: .normal)
            startStopButton.backgroundColor = UIColor.systemRed
            statusLabel.text = "🔴 CAPTURING\nMove slowly!\nFiles: 0"
            statusLabel.backgroundColor = UIColor.systemRed.withAlphaComponent(0.8)
            frameCount = 0
        } else {
            startStopButton.setTitle("START CAPTURE", for: .normal)
            startStopButton.backgroundColor = UIColor.systemGreen
            statusLabel.text = "⏸️ PAUSED\nSaved: \(frameCount/saveEvery)\nTap to restart"
            statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        }
    }

    @objc func markMove() {
        guard isCapturing else {
            statusLabel.text = "Press START first,\nthen MARK after each move"
            return
        }
        markCount += 1
        let t = Int(Date().timeIntervalSince1970 * 1000)
        let payload: [String: Any] = ["mark": markCount, "t_ms": t, "session_id": sessionID]
        if let data = try? JSONSerialization.data(withJSONObject: payload) {
            saveQueue.async { self.uploadFile(data: data, filename: "\(t)_mark.json", type: "application/json") }
        }
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
        statusLabel.text = "🔴 CAPTURING\nMarked move \(markCount)"
        print("🟠 MARK #\(markCount)")
    }

    // MARK: - ARSCNViewDelegate
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        guard let meshAnchor = anchor as? ARMeshAnchor else { return }
        print("➕ Mesh anchor added!")
        node.addChildNode(meshNode(from: meshAnchor))
    }

    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let meshAnchor = anchor as? ARMeshAnchor else { return }
        node.childNodes.forEach { $0.removeFromParentNode() }
        node.addChildNode(meshNode(from: meshAnchor))
    }

    func meshNode(from anchor: ARMeshAnchor) -> SCNNode {
        let g = anchor.geometry
        let vertexSource = SCNGeometrySource(
            buffer: g.vertices.buffer, vertexFormat: g.vertices.format,
            semantic: .vertex, vertexCount: g.vertices.count,
            dataOffset: g.vertices.offset, dataStride: g.vertices.stride)

        let normalSource = SCNGeometrySource(
            buffer: g.normals.buffer, vertexFormat: g.normals.format,
            semantic: .normal, vertexCount: g.normals.count,
            dataOffset: g.normals.offset, dataStride: g.normals.stride)

        let faceData = Data(bytesNoCopy: g.faces.buffer.contents(),
                             count: g.faces.buffer.length, deallocator: .none)
        let element = SCNGeometryElement(
            data: faceData, primitiveType: .triangles,
            primitiveCount: g.faces.count, bytesPerIndex: g.faces.bytesPerIndex)

        let geo = SCNGeometry(sources: [vertexSource, normalSource], elements: [element])

        let material = SCNMaterial()
        material.diffuse.contents = UIColor.cyan.withAlphaComponent(0.35)
        material.isDoubleSided = true
        material.blendMode = .alpha
        material.lightingModel = .physicallyBased
        material.writesToDepthBuffer = false   // avoids ugly z-fighting as the mesh updates
        geo.materials = [material]

        return SCNNode(geometry: geo)
    }

    // MARK: - ARSessionDelegate
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        totalFrames += 1
        
        if totalFrames % 60 == 0 {
            let hasDepth = frame.sceneDepth != nil
            print("📹 Frame #\(totalFrames) | Depth: \(hasDepth ? "✅" : "❌") | Tracking: \(frame.camera.trackingState) | Capturing: \(isCapturing)")
            
            // Update status if depth becomes available
            if hasDepth && !isCapturing {
                DispatchQueue.main.async {
                    self.statusLabel.text = "✅ Depth Ready!\nTap START to capture\nMove phone around"
                    self.statusLabel.backgroundColor = UIColor.systemGreen.withAlphaComponent(0.7)
                }
            }
        }
        
        if meshTest && Date().timeIntervalSince(lastMeshExport) > 1.0 {
            lastMeshExport = Date()
            exportMesh(frame)
        }
        if totalFrames % 6 == 0 && !previewBusy {  // ~10 fps live view on the Mac, capturing or not
            sendPreview(frame)
        }

        guard isCapturing else { return }

        guard case .normal = frame.camera.trackingState else {
            rejectedTracking += 1
            if totalFrames % 60 == 0 {
                print("⚠️ Tracking not normal (\(frame.camera.trackingState)) - pausing saves until it recovers")
            }
            return
        }

        guard let depthData = frame.sceneDepth else {
            if totalFrames % 120 == 0 {
                print("⚠️ Still no depth data - keep moving phone!")
            }
            return
        }

        frameCount += 1
        if frameCount % saveEvery != 0 { return }
        
        let frameNum = frameCount / saveEvery
        print("💾 Saving frame #\(frameNum)...")
        
        DispatchQueue.main.async {
            self.statusLabel.text = "🔴 CAPTURING\nMove slowly!\nFiles: \(frameNum)"
        }
        
        let capturedImage = frame.capturedImage
        let depthMap = depthData.depthMap
        let confidenceMap = depthData.confidenceMap
        let timestamp = Int(Date().timeIntervalSince1970 * 1000)
        let intrinsics = frame.camera.intrinsics
        let resolution = frame.camera.imageResolution
        let pose = frame.camera.transform  // 4x4 world-space pose, needed to fuse frames together

        saveQueue.async {
            self.uploadToMac(rgbBuffer: capturedImage, depthBuffer: depthMap, confidenceBuffer: confidenceMap,
                              timestamp: timestamp,
                              intrinsics: intrinsics, resolution: resolution, pose: pose)
        }
    }
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        print("❌ AR SESSION FAILED: \(error.localizedDescription)")
        DispatchQueue.main.async {
            self.showAlert(message: "AR failed: \(error.localizedDescription)")
        }
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        print("⚠️ Session interrupted — auto-pausing capture")
        if isCapturing {
            isCapturing = false
            DispatchQueue.main.async {
                self.startStopButton.setTitle("START CAPTURE", for: .normal)
                self.startStopButton.backgroundColor = UIColor.systemGreen
                self.statusLabel.text = "⏸️ Interrupted\nTap START to resume"
                self.statusLabel.backgroundColor = UIColor.black.withAlphaComponent(0.7)
            }
        }
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        print("✅ Session resumed — capture stays paused, tap START when ready")
    }

    // MARK: - Remote control from the Mac viewer (so nobody has to touch the phone)
    func handleRemote(_ cmd: [String: Any]) {
        if let m = cmd["mesh"] as? Bool, m != meshTest {
            meshTest = m
            applyMeshMode(m)
        }
        if let want = cmd["capture"] as? Bool, want != isCapturing {
            toggleCapture()
        }
        if let seq = cmd["mark_seq"] as? Int {
            if let last = lastMarkSeq {
                if seq > last { markMove() }
            }
            lastMarkSeq = seq   // first reply only syncs the counter; it never fires a stale mark
        }
    }

    // MARK: - ARKit mesh test (off by default; switched on from the Mac viewer)
    func applyMeshMode(_ on: Bool) {
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics = [.sceneDepth]
        if on { config.sceneReconstruction = .mesh }
        sceneView.session.run(config)          // no reset options: world tracking carries on
        print(on ? "🧱 ARKit mesh ON (exporting ~1/s)" : "🧱 ARKit mesh OFF")
    }

    /// Binary dump of every nearby ARMeshAnchor in WORLD coordinates (metres):
    ///   "MSH1" | u32 anchorCount | 16 x f32 camera transform (column-major)
    ///   per anchor: u32 vertexCount | u32 faceCount | vertexCount x (f32 x,y,z) | faceCount x (u32 a,b,c)
    func exportMesh(_ frame: ARFrame) {
        let anchors = frame.anchors.compactMap { $0 as? ARMeshAnchor }
        guard !anchors.isEmpty else { return }
        let cam = frame.camera.transform
        let camPos = SIMD3<Float>(cam.columns.3.x, cam.columns.3.y, cam.columns.3.z)
        let ts = Int(Date().timeIntervalSince1970 * 1000)
        saveQueue.async {
            var data = Data()
            func put<T>(_ v: T) { var x = v; withUnsafeBytes(of: &x) { data.append(contentsOf: $0) } }
            let near = anchors.filter { a in
                let p = SIMD3<Float>(a.transform.columns.3.x, a.transform.columns.3.y, a.transform.columns.3.z)
                return simd_distance(p, camPos) < 2.5
            }
            data.append(contentsOf: Array("MSH1".utf8))
            put(UInt32(near.count))
            for i in 0..<4 { for j in 0..<4 { put(cam[i][j]) } }
            for a in near {
                let g = a.geometry
                let vc = g.vertices.count, fc = g.faces.count
                put(UInt32(vc)); put(UInt32(fc))
                let vb = g.vertices.buffer.contents()
                for v in 0..<vc {
                    let p = vb.advanced(by: g.vertices.offset + g.vertices.stride * v).assumingMemoryBound(to: Float.self)
                    let w = a.transform * SIMD4<Float>(p[0], p[1], p[2], 1)
                    put(w.x); put(w.y); put(w.z)
                }
                let fb = g.faces.buffer.contents()
                let bpi = g.faces.bytesPerIndex
                for f in 0..<fc {
                    for k in 0..<3 {
                        let idx = fb.advanced(by: (f * 3 + k) * bpi).assumingMemoryBound(to: UInt32.self).pointee
                        put(idx)
                    }
                }
            }
            self.uploadFile(data: data, filename: "\(ts)_mesh.bin", type: "application/octet-stream")
        }
    }

    // MARK: - Live preview (viewer page on the Mac; never saved to disk)
    func sendPreview(_ frame: ARFrame) {
        guard let ip = ipAddressField.text, !ip.isEmpty,
              let url = URL(string: "http://\(ip):8000/preview") else { return }
        previewBusy = true
        let buffer = frame.capturedImage
        let capturing = isCapturing
        let saved = frameCount / saveEvery
        let marks = markCount
        let tracking: String
        switch frame.camera.trackingState {
        case .normal: tracking = "normal"
        case .notAvailable: tracking = "not available"
        case .limited: tracking = "limited"
        }
        previewQueue.async {
            let ci = CIImage(cvPixelBuffer: buffer)
            let scale = 960.0 / ci.extent.width
            let scaled = ci.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            guard let cg = self.previewContext.createCGImage(scaled, from: scaled.extent),
                  let jpeg = UIImage(cgImage: cg).jpegData(compressionQuality: 0.6) else {
                DispatchQueue.main.async { self.previewBusy = false }
                return
            }
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.timeoutInterval = 2
            request.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
            request.setValue(capturing ? "1" : "0", forHTTPHeaderField: "X-Capturing")
            request.setValue("\(saved)", forHTTPHeaderField: "X-Saved")
            request.setValue(tracking, forHTTPHeaderField: "X-Tracking")
            request.setValue("\(marks)", forHTTPHeaderField: "X-Marks")
            request.httpBody = jpeg
            URLSession.shared.dataTask(with: request) { data, _, _ in
                var cmd: [String: Any]? = nil
                if let data = data, let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    cmd = obj
                }
                DispatchQueue.main.async {
                    self.previewBusy = false
                    if let cmd = cmd { self.handleRemote(cmd) }
                }
            }.resume()
        }
    }

    // MARK: - Upload
    func uploadToMac(rgbBuffer: CVPixelBuffer, depthBuffer: CVPixelBuffer, confidenceBuffer: CVPixelBuffer?,
                      timestamp: Int,
                      intrinsics: simd_float3x3, resolution: CGSize, pose: simd_float4x4) {
        let rgbImage = CIImage(cvPixelBuffer: rgbBuffer)
        let ciContext = CIContext()
        guard let cgImage = ciContext.createCGImage(rgbImage, from: rgbImage.extent),
              let jpegData = UIImage(cgImage: cgImage).jpegData(compressionQuality: 0.9) else {
            print("❌ Failed RGB conversion")
            return
        }
        
        CVPixelBufferLockBaseAddress(depthBuffer, .readOnly)
        let width = CVPixelBufferGetWidth(depthBuffer)
        let height = CVPixelBufferGetHeight(depthBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthBuffer)
        guard let base = CVPixelBufferGetBaseAddress(depthBuffer) else {
            CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
            return
        }
        let depthData = Data(bytes: base, count: bytesPerRow * height)
        CVPixelBufferUnlockBaseAddress(depthBuffer, .readOnly)
        
        print("📦 RGB: \(jpegData.count/1024)KB, Depth: \(depthData.count/1024)KB (\(width)x\(height))")

        uploadFile(data: jpegData, filename: "\(timestamp)_rgb.jpg", type: "image/jpeg")
        uploadFile(data: depthData, filename: "\(timestamp)_depth_\(width)x\(height).bin", type: "application/octet-stream")

        // Per-pixel LiDAR confidence: 0 = low, 1 = medium, 2 = high (uint8, width*height, padding stripped)
        if let conf = confidenceBuffer {
            CVPixelBufferLockBaseAddress(conf, .readOnly)
            let cw = CVPixelBufferGetWidth(conf)
            let ch = CVPixelBufferGetHeight(conf)
            let cStride = CVPixelBufferGetBytesPerRow(conf)
            if let cBase = CVPixelBufferGetBaseAddress(conf) {
                var confData = Data(capacity: cw * ch)
                for row in 0..<ch {
                    confData.append(Data(bytes: cBase + row * cStride, count: cw))
                }
                uploadFile(data: confData, filename: "\(timestamp)_conf_\(cw)x\(ch).bin", type: "application/octet-stream")
            }
            CVPixelBufferUnlockBaseAddress(conf, .readOnly)
        }

        // Camera intrinsics (fx, fy, cx, cy) at the RGB resolution, plus the tracked
        // world-space pose (flattened 4x4, column-major) — needed to convert depth
        // pixels into XYZ and fuse multiple frames into one aligned point cloud.
        let poseFlat: [Float] = [
            pose.columns.0.x, pose.columns.0.y, pose.columns.0.z, pose.columns.0.w,
            pose.columns.1.x, pose.columns.1.y, pose.columns.1.z, pose.columns.1.w,
            pose.columns.2.x, pose.columns.2.y, pose.columns.2.z, pose.columns.2.w,
            pose.columns.3.x, pose.columns.3.y, pose.columns.3.z, pose.columns.3.w
        ]
        let intrinsicsDict: [String: Any] = [
            "fx": intrinsics.columns.0.x,
            "fy": intrinsics.columns.1.y,
            "cx": intrinsics.columns.2.x,
            "cy": intrinsics.columns.2.y,
            "rgb_width": Int(resolution.width),
            "rgb_height": Int(resolution.height),
            "session_id": sessionID,
            "pose_column_major": poseFlat
        ]
        if let intrinsicsJSON = try? JSONSerialization.data(withJSONObject: intrinsicsDict) {
            uploadFile(data: intrinsicsJSON, filename: "\(timestamp)_intrinsics.json", type: "application/json")
        }
    }
    
    func uploadFile(data: Data, filename: String, type: String) {
        guard let url = URL(string: macServerURL) else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.timeoutInterval = 30
        
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"\(filename)\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: \(type)\r\n\r\n".data(using: .utf8)!)
        body.append(data)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        request.httpBody = body
        
        DispatchQueue.main.async { self.uploadsAttempted += 1 }
        URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
            if let error = error {
                DispatchQueue.main.async { self?.uploadsFailed += 1 }
                print("❌ Upload failed [\(filename)]: \(error.localizedDescription)")
            } else if let httpResponse = response as? HTTPURLResponse {
                if httpResponse.statusCode != 200 { DispatchQueue.main.async { self?.uploadsFailed += 1 } }
                print(httpResponse.statusCode == 200 ? "✅ \(filename)" : "⚠️ HTTP \(httpResponse.statusCode)")
            }
        }.resume()
    }
}

extension ViewController: UITextFieldDelegate {
    func textFieldShouldReturn(_ textField: UITextField) -> Bool {
        textField.resignFirstResponder()
        return true
    }

    func textFieldDidEndEditing(_ textField: UITextField) {
        if let ip = textField.text, !ip.isEmpty {
            UserDefaults.standard.set(ip, forKey: "macIP")
        }
    }
}
