package com.douzero.mobile

import android.annotation.SuppressLint
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.IBinder
import android.util.DisplayMetrics
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.TextView
import kotlin.math.abs

class OverlayService : Service() {

    companion object {
        var instance: OverlayService? = null
        fun updateTip(text: String) {
            instance?.setText(text)
        }
    }

    private lateinit var windowManager: WindowManager
    private lateinit var tipView: TextView
    private lateinit var layoutParams: WindowManager.LayoutParams
    
    private var isCollapsed = false
    private var currentText = "等待识别…"
    
    // 用于拖动计算
    private var initialX = 0
    private var initialY = 0
    private var initialTouchX = 0f
    private var initialTouchY = 0f

    override fun onBind(intent: Intent?): IBinder? = null

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate() {
        super.onCreate()
        instance = this
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        
        tipView = TextView(this).apply {
            text = currentText
            setTextColor(Color.WHITE)
            textSize = 15f
            setPadding(30, 20, 30, 20)
            // 设置一个半透明的圆角背景
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#AA000000")) // 半透明黑
                cornerRadius = 20f
            }
        }

        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        else
            WindowManager.LayoutParams.TYPE_PHONE

        layoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            type,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 50
            y = 300
        }

        // 添加触摸事件（拖动 + 点击）
        tipView.setOnTouchListener { _, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    initialX = layoutParams.x
                    initialY = layoutParams.y
                    initialTouchX = event.rawX
                    initialTouchY = event.rawY
                    true
                }
                MotionEvent.ACTION_MOVE -> {
                    val dx = (event.rawX - initialTouchX).toInt()
                    val dy = (event.rawY - initialTouchY).toInt()

                    // 只有移动距离超过一定像素，才认为是拖动
                    if (abs(dx) > 10 || abs(dy) > 10) {
                        layoutParams.x = initialX + dx
                        layoutParams.y = initialY + dy
                        clampPosition()
                        windowManager.updateViewLayout(tipView, layoutParams)
                    }
                    true
                }
                MotionEvent.ACTION_UP -> {
                    val dx = event.rawX - initialTouchX
                    val dy = event.rawY - initialTouchY
                    // 如果移动距离很小，判定为“点击事件”
                    if (abs(dx) < 10 && abs(dy) < 10) {
                        toggleSize()
                    }
                    true
                }
                else -> false
            }
        }

        windowManager.addView(tipView, layoutParams)
    }

    // 限制不能拖出屏幕外
    private fun clampPosition() {
        val displayMetrics = DisplayMetrics()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val bounds = windowManager.currentWindowMetrics.bounds
            displayMetrics.widthPixels = bounds.width()
            displayMetrics.heightPixels = bounds.height()
        } else {
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getMetrics(displayMetrics)
        }
        
        val screenWidth = displayMetrics.widthPixels
        val screenHeight = displayMetrics.heightPixels
        
        // 获取当前悬浮窗的宽高
        val viewWidth = tipView.width.takeIf { it > 0 } ?: 200
        val viewHeight = tipView.height.takeIf { it > 0 } ?: 100

        // X 轴限制
        if (layoutParams.x < 0) layoutParams.x = 0
        if (layoutParams.x > screenWidth - viewWidth) layoutParams.x = screenWidth - viewWidth
        
        // Y 轴限制
        if (layoutParams.y < 0) layoutParams.y = 0
        if (layoutParams.y > screenHeight - viewHeight) layoutParams.y = screenHeight - viewHeight
    }

    // 切换 展开/折叠 状态
    private fun toggleSize() {
        isCollapsed = !isCollapsed
        if (isCollapsed) {
            // 变成一个小按钮
            tipView.text = "AI"
            tipView.setPadding(30, 30, 30, 30)
        } else {
            // 变回完整面板
            tipView.text = currentText
            tipView.setPadding(30, 20, 30, 20)
        }
        windowManager.updateViewLayout(tipView, layoutParams)
    }

    fun setText(t: String) {
        currentText = t
        // 只有在展开状态下才更新文字，折叠状态保持 "AI"
        if (!isCollapsed) {
            tipView.post { tipView.text = currentText }
        }
    }

    override fun onDestroy() {
        instance = null
        if (this::tipView.isInitialized) windowManager.removeView(tipView)
        super.onDestroy()
    }
}
