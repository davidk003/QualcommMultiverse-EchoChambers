# Mobile (Android) — secondary audio channel

**Status: scaffolded, not built or run on a real device.** No Android
SDK/Gradle toolchain or physical phone was available in the environment this
was authored in. See the blocker note in `../../ARCHITECTURE.md`.

| File | What it is |
|---|---|
| `AudioCaptureClient.kt` | Hand-written. Captures PCM via `AudioRecord` and streams frames to the fusion hub over an OkHttp WebSocket — this is the phone's actual role per the architecture (second physical vantage point for consensus/cross-correlation). |
| `InferenceEngine.kt` | Raw `mcp__quad__generate_code(platform="android", sdk="qnn")` output, unedited. Kept as a QUAD-standard record; **not used** by the current design — see its docstring. |

## Gradle dependencies (not yet wired into a build.gradle.kts — no project scaffold exists)

```
implementation("com.squareup.okhttp3:okhttp:4.12.0")
implementation("org.json:json:20240303")
implementation("androidx.core:core-ktx:1.13.1")
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
```

`AndroidManifest.xml` needs `<uses-permission android:name="android.permission.RECORD_AUDIO"/>`
plus a runtime permission request (`AudioCaptureClient.requestPermission`).

## Before trusting this on a real device

1. Create the actual Android Studio project (this scaffold is source files
   only, not a buildable Gradle project) and wire these two files in.
2. `adb devices` must show the target phone.
3. Point `hubUrl` at the X-Elite host's fusion hub address (see
   `run_echo_chamber.py --serve-fusion-hub`).
