// Read-only window diagnostics for explicitly selected processes. This helper
// neither activates applications nor requests screen-recording permissions.
// Build: xcrun swiftc -O eng/validation/macos-window-visibility.swift -o artifacts/macos-window-visibility
// Usage: artifacts/macos-window-visibility --pid 12345 [--pid 12346] [--include-titles]
import AppKit
import CoreGraphics
import Foundation

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(2)
}

func rectJSON(_ rect: CGRect) -> [String: Double] {
    ["x": rect.origin.x, "y": rect.origin.y,
     "width": rect.size.width, "height": rect.size.height]
}

func activationPolicyName(_ policy: NSApplication.ActivationPolicy) -> String {
    switch policy {
    case .regular: return "regular"
    case .accessory: return "accessory"
    case .prohibited: return "prohibited"
    @unknown default: return "unknown"
    }
}

let arguments = Array(CommandLine.arguments.dropFirst())
var pids: [pid_t] = []
var includeTitles = false
var index = 0
while index < arguments.count {
    switch arguments[index] {
    case "--pid":
        index += 1
        guard index < arguments.count else { fail("--pid requires a process ID or 'self'.") }
        let candidate = arguments[index] == "self" ? getpid() : Int32(arguments[index])
        guard let pid = candidate, pid > 0 else { fail("--pid must be a positive process ID or 'self'.") }
        if !pids.contains(pid) { pids.append(pid) }
        guard pids.count <= 32 else { fail("At most 32 process IDs may be inspected.") }
    case "--include-titles":
        includeTitles = true
    case "--help", "-h":
        print("Usage: macos-window-visibility --pid PID|self [--pid PID] [--include-titles]")
        print("Reports display geometry and selected processes' application/window metadata as JSON.")
        print("Titles are omitted by default; no screenshots or application activation are performed.")
        exit(0)
    default:
        fail("Unknown argument: \(arguments[index])")
    }
    index += 1
}
guard !pids.isEmpty else { fail("Provide at least one --pid; unrestricted application enumeration is not supported.") }

var displayIDs = [CGDirectDisplayID](repeating: 0, count: 64)
var displayCount: UInt32 = 0
let displayResult = CGGetActiveDisplayList(UInt32(displayIDs.count), &displayIDs, &displayCount)
guard displayResult == .success else { fail("CGGetActiveDisplayList failed: \(displayResult.rawValue)") }
let displays: [[String: Any]] = displayIDs.prefix(Int(displayCount)).map { display in
    ["displayID": display,
     "bounds": rectJSON(CGDisplayBounds(display)),
     "isMain": CGDisplayIsMain(display) != 0,
     "isBuiltin": CGDisplayIsBuiltin(display) != 0]
}

// CoreGraphics has no PID-specific list API. Filter by owner PID before reading
// any other window metadata, especially optional titles.
guard let windowList = CGWindowListCopyWindowInfo(.optionAll, kCGNullWindowID) as? [[String: Any]] else {
    fail("CGWindowListCopyWindowInfo did not return window metadata.")
}
let selectedPids = Set(pids)
let selectedWindows = windowList.filter { window in
    guard let owner = window[kCGWindowOwnerPID as String] as? NSNumber else { return false }
    return selectedPids.contains(owner.int32Value)
}

let processes: [[String: Any]] = pids.map { pid in
    var item: [String: Any] = ["pid": pid]
    if let app = NSRunningApplication(processIdentifier: pid) {
        item["application"] = [
            "activationPolicy": activationPolicyName(app.activationPolicy),
            "activationPolicyRawValue": app.activationPolicy.rawValue,
            "isHidden": app.isHidden,
            "isActive": app.isActive,
            "isTerminated": app.isTerminated,
            "bundleURL": app.bundleURL?.path as Any? ?? NSNull(),
            "bundleIdentifier": app.bundleIdentifier as Any? ?? NSNull()
        ] as [String: Any]
    } else {
        item["application"] = NSNull()
    }
    let owned = selectedWindows.filter {
        ($0[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value == pid
    }
    item["windowCount"] = owned.count
    item["windowsTruncated"] = owned.count > 256
    item["windows"] = owned.prefix(256).map { window -> [String: Any] in
        var entry: [String: Any] = [
            "windowID": window[kCGWindowNumber as String] ?? NSNull(),
            "layer": window[kCGWindowLayer as String] ?? NSNull(),
            "onScreen": window[kCGWindowIsOnscreen as String] ?? NSNull(),
            "alpha": window[kCGWindowAlpha as String] ?? NSNull()
        ]
        if let rawBounds = window[kCGWindowBounds as String] as? NSDictionary,
           let bounds = CGRect(dictionaryRepresentation: rawBounds) {
            entry["bounds"] = rectJSON(bounds)
            entry["intersectsActiveDisplay"] = displayIDs.prefix(Int(displayCount)).contains {
                CGDisplayBounds($0).intersects(bounds)
            }
        } else {
            entry["bounds"] = NSNull()
            entry["intersectsActiveDisplay"] = NSNull()
        }
        if includeTitles { entry["title"] = window[kCGWindowName as String] ?? NSNull() }
        return entry
    }
    return item
}

let result: [String: Any] = [
    "schemaVersion": 1,
    "capturedAt": ISO8601DateFormatter().string(from: Date()),
    "titlesIncluded": includeTitles,
    "displays": displays,
    "processes": processes
]
do {
    let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
} catch {
    fail("Could not serialize window metadata: \(error)")
}
