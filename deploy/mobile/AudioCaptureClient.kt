package com.qualcomm.quad.echochamber

/**
 * Mobile secondary audio channel for Echo Chamber.
 *
 * *** NOT BUILT OR RUN ON A REAL DEVICE -- see the blocker note in
 * ARCHITECTURE.md. No Android SDK/Gradle toolchain or physical phone was
 * available in the environment this was authored in; this has been reviewed
 * for API correctness (AudioRecord + OkHttp WebSocket usage matches current
 * Android APIs) but not compiled or run. Stop and hand this to a real device
 * + `adb devices` connection before trusting it. ***
 *
 * Role (per the proposal architecture): a second physical vantage point.
 * Captures raw PCM at [SAMPLE_RATE_HZ] and streams frames to the X-Elite
 * fusion hub, which performs the actual cross-correlation / consensus check
 * against the primary (UNO Q) and host-mic streams within a 200 ms window --
 * the phone does not run the classifier itself, matching the same
 * capture-only role as the Arduino UNO Q's capture_agent.py.
 *
 * Requires RECORD_AUDIO permission (request at runtime, not just in the
 * manifest) and a network security config permitting cleartext ws:// to the
 * host's LAN address, or use wss:// with a real certificate in production.
 */

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONArray
import org.json.JSONObject

class AudioCaptureClient(
    private val hubUrl: String,
    private val deviceId: String = "mobile-1",
) {
    companion object {
        private const val TAG = "EchoChamberCapture"
        const val SAMPLE_RATE_HZ = 48_000
        const val FRAME_SAMPLES = 1024 // matches echo_chamber.N_FFT on the host
    }

    private var audioRecord: AudioRecord? = null
    private var webSocket: WebSocket? = null
    @Volatile private var running = false

    fun hasPermission(context: android.content.Context): Boolean =
        ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED

    fun requestPermission(activity: android.app.Activity, requestCode: Int) {
        ActivityCompat.requestPermissions(activity, arrayOf(Manifest.permission.RECORD_AUDIO), requestCode)
    }

    fun start(scope: CoroutineScope) {
        val client = OkHttpClient()
        val request = Request.Builder().url(hubUrl).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(ws: WebSocket, response: okhttp3.Response) {
                val hello = JSONObject().apply {
                    put("type", "hello")
                    put("device_id", deviceId)
                    put("device_kind", "mobile")
                    put("sample_rate_hz", SAMPLE_RATE_HZ)
                }
                ws.send(hello.toString())
                beginCapture(scope, ws)
            }

            override fun onFailure(ws: WebSocket, t: Throwable, response: okhttp3.Response?) {
                Log.e(TAG, "fusion hub connection failed: ${t.message}")
            }
        })
    }

    @Suppress("MissingPermission") // caller must verify hasPermission() first
    private fun beginCapture(scope: CoroutineScope, ws: WebSocket) {
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE_HZ, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT,
        )
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.UNPROCESSED, SAMPLE_RATE_HZ,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT,
            maxOf(minBuf, FRAME_SAMPLES * 4),
        )
        running = true
        audioRecord?.startRecording()

        scope.launch(Dispatchers.IO) {
            val buffer = FloatArray(FRAME_SAMPLES)
            while (running) {
                val read = audioRecord?.read(buffer, 0, FRAME_SAMPLES, AudioRecord.READ_BLOCKING) ?: -1
                if (read <= 0) continue
                val samples = JSONArray()
                for (i in 0 until read) samples.put(buffer[i].toDouble())
                val msg = JSONObject().apply {
                    put("type", "audio_frame")
                    put("device_id", deviceId)
                    put("t_capture", System.currentTimeMillis() / 1000.0)
                    put("samples", samples)
                }
                ws.send(msg.toString())
            }
        }
    }

    fun stop() {
        running = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        webSocket?.close(1000, "client stopped")
    }
}
