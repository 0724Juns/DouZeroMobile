package com.douzero.mobile

import android.annotation.SuppressLint
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.IBinder
import android.util.DisplayMetrics
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.TextView
import kotlin.math.abs

class OverlayService : Service() {

    companion object {
        var instance: OverlayService? = null
        fun updateTip(text: String, boxes: List<BoundingBox> = emptyList()) {
            instance?.setText(text, boxes)
        }
    }

    private lateinit var windowManager: WindowManager
    private lateinit var tipView: TextView
    private lateinit var boxView: BoxDrawView
    private lateinit var boxLayoutParams: WindowManager.LayoutParams
    private lateinit var tipLayoutParams: WindowManager.LayoutParams
    
    private var isCollapsed = false
    private var currentText = "等待识别…"

    override fun onBind(intent: Intent?): IBinder? = null

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate() {
        super.onCreate()
        instance = this
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager

        // 初始化一个全屏透明层用来画框
        boxView = BoxDrawView(this)
        boxLayoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY else WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
            WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT
        )
        windowManager.addView(boxView, boxLayoutParams)

        // 初始化提示文字
        tipView = TextView(this).apply {
            text = currentText
            setTextColor(Color.WHITE)
            textSize = 15f
            setPadding(30, 20, 30, 20)
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#DD000000"))
                cornerRadius = 20f
            }
        }

        tipLayoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY else WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 50
            y = 300
        }

        tipView.setOnTouchListener(object : View.OnTouchListener {
            private var initialX = 0
            private var initialY = 0
            private var initialTouchX = 0f
            private var initialTouchY = 0f

            override fun onTouch(v: View?, event: MotionEvent): Boolean {
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        initialX = tipLayoutParams.x
                        initialY = tipLayoutParams.y
                        initialTouchX = event.rawX
                        initialTouchY = event.rawY
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val dx = (event.rawX - initialTouchX).toInt()
                        val dy = (event.rawY - initialTouchY).toInt()
                        if (abs(dx) > 10 || abs(dy) > 10) {
                            tipLayoutParams.x = initialX + dx
                            tipLayoutParams.y = initialY + dy
                            clampPosition()
                            windowManager.updateViewLayout(tipView, tipLayoutParams)
                        }
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        val dx = event.rawX - initialTouchX
                        val dy = event.rawY - initialTouchY
                        if (abs(dx) < 10 && abs(dy) < 10) {
                            toggleSize()
                        }
                        return true
                    }
                }
                return false
            }
        })

        windowManager.addView(tipView, tipLayoutParams)
    }

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
        val w = tipView.width.takeIf { it > 0 } ?: 200
        val h = tipView.height.takeIf { it > 0 } ?: 100
        if (tipLayoutParams.x < 0) tipLayoutParams.x = 0
        if (tipLayoutParams.x > displayMetrics.widthPixels - w) tipLayoutParams.x = displayMetrics.widthPixels - w
        if (tipLayoutParams.y < 0) tipLayoutParams.y = 0
        if (tipLayoutParams.y > displayMetrics.heightPixels - h) tipLayoutParams.y = displayMetrics.heightPixels - h
    }

    private fun toggleSize() {
        isCollapsed = !isCollapsed
        if (isCollapsed) {
            tipView.text = "AI"
            tipView.setPadding(30, 30, 30, 30)
        } else {
            tipView.text = currentText
            tipView.setPadding(30, 20, 30, 20)
        }
        windowManager.updateViewLayout(tipView, tipLayoutParams)
    }

    fun setText(t: String, boxes: List<BoundingBox>) {
        currentText = t
        if (!isCollapsed) {
            tipView.post { tipView.text = currentText }
        }
        // 更新画框层
        boxView.post {
            boxView.updateBoxes(boxes)
        }
    }

    override fun onDestroy() {
        instance = null
        if (this::tipView.isInitialized) windowManager.removeView(tipView)
        if (this::boxView.isInitialized) windowManager.removeView(boxView)
        super.onDestroy()
    }
}

// 自定义 View，负责在屏幕上画红色的框
@SuppressLint("ViewConstructor")
class BoxDrawView(context: Context) : View(context) {
    private var boxes: List<BoundingBox> = emptyList()
    private val paint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 5f
        isAntiAlias = true
    }
    private val textPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 40f
        isAntiAlias = true
        style = Paint.Style.FILL
    }

    fun updateBoxes(newBoxes: List<BoundingBox>) {
        boxes = newBoxes
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        for (box in boxes) {
            canvas.drawRect(box.x, box.y, box.x + box.w, box.y + box.h, paint)
            canvas.drawText(box.label, box.x, box.y - 10, textPaint)
        }
    }
}
