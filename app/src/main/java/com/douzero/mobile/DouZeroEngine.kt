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

// 增加一个数据类，用来保存画框的坐标
data class BoundingBox(
    val x: Float,
    val y: Float,
    val w: Float,
    val h: Float,
    val label: String
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
    val boxes: List<BoundingBox> = emptyList() // 新增：保存要画的框
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

            // 存储所有候选框
            data class TempBox(val cx: Float, val cy: Float, val w: Float, val h: Float, val conf: Float, val classId: Int)
            val candidates = mutableListOf<TempBox>()

            for (i in 0 until numBoxes) {
                var maxProb = -999f
                var classId = -1
                
                for (c in 0 until numClasses) {
                    val probIndex = if (isChannelFirst) (4 + c) * numBoxes + i else i * (4 + numClasses) + (4 + c)
                    var prob = outputArray[probIndex]
                    // 如果有负数，说明没做 Sigmoid，手动激活一下
                    if (prob < 0f || prob > 1f) prob = 1f / (1f + exp(-prob))
                    
                    if (prob > maxProb) {
                        maxProb = prob
                        classId = c
                    }
                }

                if (maxProb > 0.25f) {
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

            // 执行 NMS (非极大值抑制) 去重
            val finalBoxes = mutableListOf<BoundingBox>()
            val myHand = mutableListOf<Int>()
            candidates.sortByDescending { it.conf }
            
            val selected = mutableListOf<TempBox>()
            for (cand in candidates) {
                var keep = true
                for (sel in selected) {
                    // 计算 IoU (重叠度)
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
                // 坐标还原到屏幕尺寸
                val realX = (box.cx - box.w/2) * screen.width
                val realY = (box.cy - box.h/2) * screen.height
                val realW = box.w * screen.width
                val realH = box.h * screen.height
                
                val labelName = cardNames.getOrElse(box.classId) { "?" }
                finalBoxes.add(BoundingBox(realX, realY, realW, realH, labelName))
                
                // 如果这个框在屏幕下方 (Y > 屏幕高度的一半)，认为是手牌
                if (box.cy > 0.5f) {
                    myHand.add(box.classId)
                }
            }

            val debugStr = "找到${finalBoxes.size}张牌"

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

// DouZeroEngine 和 ActionDecoder 保持不变
class DouZeroEngine(private val context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val sessions = mutableMapOf<Role, OrtSession>()

    init {
        listOf(
            Role.LANDLORD to "landlord.onnx",
            Role.LANDLORD_UP to "landlord_up.onnx",
            Role.LANDLORD_DOWN to "landlord_down.onnx"
        ).forEach { (role, name) ->
            try {
                val modelBytes = context.assets.open(name).readBytes()
                sessions[role] = env.createSession(modelBytes, OrtSession.SessionOptions())
            } catch (e: Exception) { }
        }
    }

    fun hasModel(): Boolean = sessions.isNotEmpty()

    fun infer(state: GameState): FloatArray {
        val session = sessions[state.myRole] ?: return FloatArray(0)
        return try {
            val handTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(encodeHand(state.myHand)), longArrayOf(1, 4, 15))
            val inputs = mapOf("hand_cards" to handTensor)
            session.run(inputs).use { result -> (result.get(0) as OnnxTensor).floatBuffer.array() }
        } catch (_: Exception) { FloatArray(0) }
    }

    private fun encodeHand(hand: List<Int>): FloatArray {
        val flatArray = FloatArray(4 * 15) 
        val counts = IntArray(15)
        hand.forEach { if (it in 0..14) counts[it]++ }
        for (v in 0 until 15) {
            for (r in 0 until counts[v].coerceAtMost(4)) { flatArray[r * 15 + v] = 1f }
        }
        return flatArray
    }

    fun close() { sessions.values.forEach { it.close() }; env.close() }
}

object ActionDecoder {
    data class Suggestion(val action: String, val cards: List<Int>)
    fun suggest(state: GameState, qValues: FloatArray): Suggestion {
        if (state.isLandlordTurnToCall) return decideCall(state.myHand)
        if (state.isDoublingPhase) return decideDouble(state.myHand)
        return decidePlay(state)
    }
    private fun decideCall(hand: List<Int>): Suggestion {
        val s = handStrength(hand)
        return when { s >= 8 -> Suggestion("抢地主", emptyList()); s >= 5 -> Suggestion("考虑抢地主", emptyList()); else -> Suggestion("不抢", emptyList()) }
    }
    private fun decideDouble(hand: List<Int>): Suggestion { return if (handStrength(hand) >= 7) Suggestion("加倍", emptyList()) else Suggestion("不加倍", emptyList()) }
    private fun handStrength(hand: List<Int>): Int {
        var score = 0
        val counts = IntArray(15)
        hand.forEach { if (it in 0..14) counts[it]++ }
        if (counts[13] > 0) score += 3; if (counts[14] > 0) score += 4; score += counts[12] * 2
        for (v in 0..12) if (counts[v] == 4) score += 5
        return score
    }
    private fun decidePlay(state: GameState): Suggestion {
        val hand = state.myHand.sorted()
        if (hand.isEmpty()) return Suggestion("无牌可出", emptyList())
        val lastMove = state.lastMoves.lastOrNull { it.isNotEmpty() }
        if (lastMove == null) return Suggestion("出 ${cardName(hand.first())}", listOf(hand.first()))
        if (lastMove.size == 1) {
            val beat = hand.firstOrNull { it > lastMove[0] }
            if (beat != null) return Suggestion("出 ${cardName(beat)}", listOf(beat))
        }
        return Suggestion("过牌", emptyList())
    }
    private fun cardName(v: Int): String = when (v) { 12 -> "2"; 13 -> "小王"; 14 -> "大王"; else -> listOf("3","4","5","6","7","8","9","10","J","Q","K")[v] }
}
