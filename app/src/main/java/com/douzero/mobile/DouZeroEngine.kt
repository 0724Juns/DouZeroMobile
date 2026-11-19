package com.douzero.mobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.exp

enum class Role { LANDLORD, LANDLORD_UP, LANDLORD_DOWN }

data class BoundingBox(
    val x: Float, val y: Float, val w: Float, val h: Float, val label: String
)

data class GameState(
    val myRole: Role,
    val myHand: List<Int>,
    val lastMoves: List<List<Int>>,
    val otherCardCounts: List<Int>,
    val bombCount: Int,
    val isLandlordTurnToCall: Boolean = false,
    val isDoublingPhase: Boolean = false,
    val debugInfo: String = "",
    val boxes: List<BoundingBox> = emptyList()
)

interface CardRecognizer {
    fun recognize(screen: Bitmap): GameState?
}

class YoloCardRecognizer(private val context: Context) : CardRecognizer {
    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var loadError: String = "未知错误"
    private val inputWidth = 960
    private val inputHeight = 608

    init {
        try {
            val f = File(context.filesDir, "best.onnx")
            if (f.exists()) { f.delete() }
            val modelBytes = context.assets.open("best.onnx").readBytes()
            session = env.createSession(modelBytes, OrtSession.SessionOptions())
        } catch (e: Exception) {
            loadError = e.message ?: "文件读取失败"
        }
    }

    override fun recognize(screen: Bitmap): GameState? {
        val s = session ?: return GameState(Role.LANDLORD, emptyList(), emptyList(), emptyList(), 0, debugInfo = "报错:$loadError")
        
        try {
            val resized = Bitmap.createScaledBitmap(screen, inputWidth, inputHeight, true)
            val inputBuffer = convertBitmapToFloatBuffer(resized, normalize = true)

            val inputTensor = OnnxTensor.createTensor(
                env, inputBuffer,
                longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
            )
            
            val inputName = s.inputNames.iterator().next()
            val result = s.run(mapOf(inputName to inputTensor))
            val output = result.get(0) as OnnxTensor
            val shape = output.info.shape
            val outputArray = output.floatBuffer.array()
            result.close()
            inputTensor.close()

            if (shape.size != 3) return GameState(Role.LANDLORD, emptyList(), emptyList(), emptyList(), 0, debugInfo = "维度异常:$shape")

            val dim1 = shape[1].toInt()
            val dim2 = shape[2].toInt()
            val numClasses = 15
            var numBoxes = 0
            var isChannelFirst = true

            if (dim1 == 4 + numClasses) {
                numBoxes = dim2
                isChannelFirst = true
            } else if (dim2 == 4 + numClasses) {
                numBoxes = dim1
                isChannelFirst = false
            } else {
                 return GameState(Role.LANDLORD, emptyList(), emptyList(), emptyList(), 0, debugInfo = "无法识别维度:$shape")
            }

            data class TempBox(val cx: Float, val cy: Float, val w: Float, val h: Float, val conf: Float, val classId: Int)
            val candidates = mutableListOf<TempBox>()

            for (i in 0 until numBoxes) {
                var maxProb = -999f
                var classId = -1
                
                for (c in 0 until numClasses) {
                    val probIndex = if (isChannelFirst) (4 + c) * numBoxes + i else i * (4 + numClasses) + (4 + c)
                    var prob = outputArray[probIndex]
                    if (prob < 0f || prob > 1f) prob = 1f / (1f + exp(-prob))
                    
                    if (prob > maxProb) {
                        maxProb = prob
                        classId = c
                    }
                }

                // 【修改】：把门槛降低回 0.1，让模型大胆一点
                if (maxProb > 0.1f) {
                    val cxIndex = if (isChannelFirst) 0 * numBoxes + i else i * (4 + numClasses) + 0
                    val cyIndex = if (isChannelFirst) 1 * numBoxes + i else i * (4 + numClasses) + 1
                    val wIndex = if (isChannelFirst) 2 * numBoxes + i else i * (4 + numClasses) + 2
                    val hIndex = if (isChannelFirst) 3 * numBoxes + i else i * (4 + numClasses) + 3
                    
                    candidates.add(TempBox(
                        outputArray[cxIndex], outputArray[cyIndex], 
                        outputArray[wIndex], outputArray[hIndex], 
                        maxProb, classId
                    ))
                }
            }

            val finalBoxes = mutableListOf<BoundingBox>()
            val myHand = mutableListOf<Int>()
            candidates.sortByDescending { it.conf }
            
            val selected = mutableListOf<TempBox>()
            for (cand in candidates) {
                var keep = true
                for (sel in selected) {
                    val x1 = max(cand.cx - cand.w/2, sel.cx - sel.w/2)
                    val y1 = max(cand.cy - cand.h/2, sel.cy - sel.h/2)
                    val x2 = min(cand.cx + cand.w/2, sel.cx + sel.w/2)
                    val y2 = min(cand.cy + cand.h/2, sel.cy + sel.h/2)
                    val intersection = max(0f, x2 - x1) * max(0f, y2 - y1)
                    val area1 = cand.w * cand.h
                    val area2 = sel.w * sel.h
                    val iou = intersection / (area1 + area2 - intersection)
                    
                    if (iou > 0.45f) {
                        keep = false
                        break
                    }
                }
                if (keep) selected.add(cand)
            }

            val cardNames = listOf("3","4","5","6","7","8","9","10","J","Q","K","A","2","小王","大王")

            for (box in selected) {
                val realX = (box.cx - box.w/2) * screen.width
                val realY = (box.cy - box.h/2) * screen.height
                val realW = box.w * screen.width
                val realH = box.h * screen.height
                
                val labelName = cardNames.getOrElse(box.classId) { "?" }
                finalBoxes.add(BoundingBox(realX, realY, realW, realH, labelName))
                
                if (box.cy > 0.5f) {
                    myHand.add(box.classId)
                }
            }

            // 把原始候选框数量和最终去重后的数量都打印出来，方便观察
            val debugStr = "去重前:${candidates.size} 去重后:${finalBoxes.size}"

            if (myHand.isEmpty() && finalBoxes.isEmpty()) {
                return GameState(Role.LANDLORD, emptyList(), emptyList(), emptyList(), 0, debugInfo = "没找到牌")
            }

            return GameState(
                Role.LANDLORD, myHand, 
                listOf(emptyList(), emptyList(), emptyList()), 
                listOf(17, 17), 0, debugInfo = debugStr, boxes = finalBoxes
            )
        } catch (e: Exception) {
            return GameState(Role.LANDLORD, emptyList(), emptyList(), emptyList(), 0, debugInfo = "推理错:${e.message}")
        }
    }

    private fun convertBitmapToFloatBuffer(bitmap: Bitmap, normalize: Boolean): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val floatBuffer = FloatBuffer.allocate(3 * width * height)
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val rChannel = FloatArray(width * height)
        val gChannel = FloatArray(width * height)
        val bChannel = FloatArray(width * height)
        val scale = if (normalize) 1.0f / 255.0f else 1.0f

        for (i in pixels.indices) {
            val p = pixels[i]
            rChannel[i] = (p shr 16 and 0xFF) * scale
            gChannel[i] = (p shr 8 and 0xFF) * scale
            bChannel[i] = (p and 0xFF) * scale
        }
        floatBuffer.put(rChannel)
        floatBuffer.put(gChannel)
        floatBuffer.put(bChannel)
        floatBuffer.rewind()
        return floatBuffer
    }
}
