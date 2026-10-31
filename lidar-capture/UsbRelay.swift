//
//  UsbRelay.swift
//  lidar-capture
//
//  Wired mode.  Over a USB cable the Mac can reach the phone, but the phone cannot reach the Mac, so the usual
//  "phone POSTs to the Mac" does not work.  Instead the phone keeps its outgoing requests in a queue and the Mac
//  (scripts/usb_bridge.py, through `iproxy`) collects them from port 8765, plays them to the game server and
//  hands the answers back.  When no bridge is collecting, everything goes out over Wi-Fi exactly as before.
//
//    GET  /next        -> the next queued request: one JSON line {id, method, path, headers}, then the body
//                         (waits up to 1.5 s; 204 when nothing is queued)
//    POST /reply/<id>  -> the game server's answer to that request (status in X-Status)
//

import Foundation
import Network

final class UsbRelay {
    static let shared = UsbRelay()

    private struct Item { let id: Int; let payload: Data }
    private struct Parked { let id: Int; let conn: NWConnection }

    private let port: NWEndpoint.Port = 8765
    private let q = DispatchQueue(label: "com.lidarcapture.usbrelay")
    private var listener: NWListener?
    private var queue: [Item] = []
    private var parked: [Parked] = []
    private var pending: [Int: (Data?, Int?, Error?) -> Void] = [:]
    private var nextId = 1
    private var lastPoll = Date.distantPast
    private let maxQueued = 40

    /// True while the Mac is collecting requests (a poll is parked, or the last one was under 3 s ago).
    var isWired: Bool {
        q.sync { !parked.isEmpty || Date().timeIntervalSince(lastPoll) < 3 }
    }

    func start() {
        q.async {
            guard self.listener == nil, let l = try? NWListener(using: .tcp, on: self.port) else {
                print("⚠️ USB relay: could not listen on 8765 (Wi-Fi still works)")
                return
            }
            l.newConnectionHandler = { c in self.accept(c) }
            l.stateUpdateHandler = { state in print("🔌 USB relay \(state)") }
            l.start(queue: self.q)
            self.listener = l
        }
    }

    /// Queue a request for the Mac to pick up.  `done` gets the game server's answer, or an error after `timeout`.
    func enqueue(_ request: URLRequest, timeout: TimeInterval, done: ((Data?, Int?, Error?) -> Void)?) {
        q.async {
            let id = self.nextId; self.nextId += 1
            let url = request.url
            var path = url?.path ?? "/"
            if let query = url?.query { path += "?" + query }
            let meta: [String: Any] = ["id": id, "method": request.httpMethod ?? "GET", "path": path,
                                       "headers": request.allHTTPHeaderFields ?? [:]]
            guard var payload = try? JSONSerialization.data(withJSONObject: meta) else { return }
            payload.append(0x0A)
            payload.append(request.httpBody ?? Data())
            self.pending[id] = done ?? { _, _, _ in }
            self.queue.append(Item(id: id, payload: payload))
            while self.queue.count > self.maxQueued {            // the Mac fell behind: drop the oldest
                let old = self.queue.removeFirst()
                self.finish(old.id, nil, nil, NSError(domain: "UsbRelay", code: 1, userInfo: [NSLocalizedDescriptionKey: "queue full"]))
            }
            self.flush()
            self.q.asyncAfter(deadline: .now() + timeout) {
                self.queue.removeAll { $0.id == id }
                self.finish(id, nil, nil, NSError(domain: "UsbRelay", code: 2, userInfo: [NSLocalizedDescriptionKey: "no answer over USB"]))
            }
        }
    }

    // MARK: - queue handling (always on `q`)

    private func finish(_ id: Int, _ data: Data?, _ status: Int?, _ error: Error?) {
        guard let cb = pending.removeValue(forKey: id) else { return }
        cb(data, status, error)
    }

    private func flush() {
        while let p = parked.first, let item = queue.first {
            parked.removeFirst(); queue.removeFirst()
            respond(p.conn, status: "200 OK", body: item.payload)
        }
    }

    // MARK: - a very small HTTP server

    private func accept(_ c: NWConnection) {
        c.start(queue: q)
        receive(c, Data())
    }

    private func receive(_ c: NWConnection, _ have: Data) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, closed, error in
            guard let self = self else { return }
            var buf = have
            if let d = data { buf.append(d) }
            if let end = buf.range(of: Data("\r\n\r\n".utf8)) {
                let head = String(decoding: buf[buf.startIndex..<end.lowerBound], as: UTF8.self)
                var length = 0
                for line in head.components(separatedBy: "\r\n") where line.lowercased().hasPrefix("content-length:") {
                    length = Int(line.dropFirst(15).trimmingCharacters(in: .whitespaces)) ?? 0
                }
                if buf.count - end.upperBound >= length {
                    let body = buf.subdata(in: end.upperBound..<(end.upperBound + length))
                    self.handle(c, head: head, body: body)
                    return
                }
            }
            if closed || error != nil { c.cancel(); return }
            self.receive(c, buf)
        }
    }

    private func handle(_ c: NWConnection, head: String, body: Data) {
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines[0].split(separator: " ")
        let path = parts.count > 1 ? String(parts[1]) : "/"
        if path == "/next" {
            lastPoll = Date()
            if !queue.isEmpty {
                respond(c, status: "200 OK", body: queue.removeFirst().payload)
                return
            }
            let id = nextId; nextId += 1
            parked.append(Parked(id: id, conn: c))
            q.asyncAfter(deadline: .now() + 1.5) {
                guard let i = self.parked.firstIndex(where: { $0.id == id }) else { return }
                self.parked.remove(at: i)
                self.lastPoll = Date()
                self.respond(c, status: "204 No Content", body: Data())
            }
        } else if path.hasPrefix("/reply/"), let id = Int(path.dropFirst(7)) {
            var status = 200
            for line in lines where line.lowercased().hasPrefix("x-status:") {
                status = Int(line.dropFirst(9).trimmingCharacters(in: .whitespaces)) ?? 200
            }
            finish(id, body, status, nil)
            respond(c, status: "200 OK", body: Data())
        } else {
            respond(c, status: "404 Not Found", body: Data())
        }
    }

    private func respond(_ c: NWConnection, status: String, body: Data) {
        var out = Data("HTTP/1.1 \(status)\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n".utf8)
        out.append(body)
        c.send(content: out, completion: .contentProcessed { _ in c.cancel() })
    }
}
