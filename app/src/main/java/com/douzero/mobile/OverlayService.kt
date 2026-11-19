package com.douzero.mobile

import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.os.Build
import android.os.IBinder
import android.view.Gravity
import android.view.WindowManager
import android.widget.TextView

class OverlayService : Service() {

    companion object {
        var instance: OverlayService? = null
        fun updateTip(text: String) {
            instance?.setText(text)
        }
    }

    private lateinit var windowManager: WindowManager
    private lateinit var tipView: TextView

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        instance = this
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        tipView = TextView(this).apply {
            text = "等待识别…"
            setTextColor(0xFFFFFFFF.toInt())
            setBackgroundColor(0xAA000000.toInt())
            textSize = 16f
            setPadding(24, 16, 24, 16)
        }

        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        else
            WindowManager.LayoutParams.TYPE_PHONE

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            type,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 20; y = 200
        }

        windowManager.addView(tipView, params)
    }

    fun setText(t: String) {
        tipView.post { tipView.text = t }
    }

    override fun onDestroy() {
        instance = null
        if (this::tipView.isInitialized) windowManager.removeView(tipView)
        super.onDestroy()
    }
}
