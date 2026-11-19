package com.douzero.mobile

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.hardware.display.DisplayManager
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.view.WindowManager
import androidx.core.app.NotificationCompat

class ScreenCaptureService : Service() {

    private lateinit var mediaProjection: MediaProjection
    private lateinit var imageReader: ImageReader
    private lateinit var virtualDisplay: android.hardware.display.VirtualDisplay
    private lateinit var recognizer: CardRecognizer
    private lateinit var engine: DouZeroEngine
    private val handler = Handler(Looper.getMainLooper())
    private var running = false

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val resultCode = intent?.getIntExtra("resultCode", -1) ?: return START_NOT_STICKY
        val data = intent.getParcelableExtra<Intent>("data") ?: return START_NOT_STICKY

        createNotificationChannel()
        val notification = NotificationCompat.Builder(this, "capture")
            .setContentTitle("DouZero 提示运行中")
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .build()
        startForeground(1, notification)

        val mpm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjection = mpm.getMediaProjection(resultCode, data)!!
        mediaProjection.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() { stopSelf() }
        }, handler)

        val metrics = DisplayMetrics()
        (getSystemService(Context.WINDOW_SERVICE) as WindowManager).defaultDisplay.getMetrics(metrics)
        val w = metrics.widthPixels
        val h = metrics.heightPixels

        imageReader = ImageReader.newInstance(w, h, android.graphics.PixelFormat.RGBA_8888, 2)
        virtualDisplay = mediaProjection.createVirtualDisplay(
            "douzero-capture", w, h, metrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader.surface, null, handler
        )

        recognizer = YoloCardRecognizer(applicationContext)
        engine = DouZeroEngine(applicationContext)
        
        running = true
        loop()
        return START_STICKY
    }

    private fun loop() {
        if (!running) return
        handler.postDelayed({
            captureAndInfer()
            loop()
        }, 1500) 
    }

    private fun captureAndInfer() {
        val image = imageReader.acquireLatestImage() ?: return
        try {
            val buffer = image.planes[0].buffer
            val bitmap = Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
            bitmap.copyPixelsFromBuffer(buffer)

            val state = recognizer.recognize(bitmap)
            
            if (state == null) {
                OverlayService.updateTip("状态为空", emptyList())
                return
            }

            // 无论有没有识别到，都把牌的序号转成名称
            val handNames = state.myHand.map { cardName(it) }.joinToString(" ")

            if (state.myHand.isEmpty()) {
                OverlayService.updateTip("未识别到手牌\n${state.debugInfo}", state.boxes)
                return
            }

            val q = engine.infer(state)
            val suggestion = ActionDecoder.suggest(state, q)
            
            // 如果 AI 大脑没加载出来，提示一下
            val modelNote = if (engine.hasModel()) "" else "（未加载AI，启发式）"
            
            // 【终极展示】：把手牌铺开，下面跟上建议
            val showText = "手牌: $handNames\n建议: $modelNote${suggestion.action}"
            OverlayService.updateTip(showText, state.boxes)
            
        } catch (e: Exception) {
            OverlayService.updateTip("报错: ${e.message}", emptyList())
        } finally {
            image.close()
        }
    }

    // 牌面翻译器
    private fun cardName(v: Int): String = when (v) {
        0 -> "3"; 1 -> "4"; 2 -> "5"; 3 -> "6"; 4 -> "7"
        5 -> "8"; 6 -> "9"; 7 -> "10"; 8 -> "J"; 9 -> "Q"
        10 -> "K"; 11 -> "A"; 12 -> "2"; 13 -> "小王"; 14 -> "大王"
        else -> "?"
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel("capture", "屏幕采集", NotificationManager.IMPORTANCE_LOW)
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
    }

    override fun onDestroy() {
        running = false
        virtualDisplay.release()
        imageReader.close()
        mediaProjection.stop()
        engine.close()
        super.onDestroy()
    }
}
