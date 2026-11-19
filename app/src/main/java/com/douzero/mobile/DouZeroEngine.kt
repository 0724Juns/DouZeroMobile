package com.douzero.mobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.nio.FloatBuffer

enum class Role { LANDLORD, LANDLORD_UP, LANDLORD_DOWN }

data class GameState(
    val myRole: Role,
    val myHand: List<Int>,
    val lastMoves: List<List<Int>>,
    val otherCardCounts: List<Int>,
    val bombCount: Int,
    val isLandlordTurnToCall: Boolean = false,
    val isDoublingPhase: Boolean = false,
    val debugInfo: String = ""
)

interface CardRecognizer {
    fun recognize(screen: Bitmap): GameState?
}

class YoloCardRecognizer(private val context: Context) : CardRecognizer {
    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private var loadError: String = "未知错误"
    private var inputWidth = 640
    private var inputHeight = 640

    init {
        try {
            // 【关键修复】：强制清理手机里的旧模型缓存，每次都重新从 assets 读
            val f = File(context.filesDir, "best.onnx")
            if (f.exists()) { f.delete() }
            
            val modelBytes = context.assets.open("best.onnx").readBytes()
            session = env.createSession(modelBytes, OrtSession.SessionOptions())
            
            // 【自动探测】：读取模型真正想要的尺寸
            val inputName = session!!.inputNames.iterator().next()
            val tensorInfo = session!!.inputInfo[inputName] as TensorInfo
            val shape = tensorInfo.shape
            if (shape.size == 4) {
                inputHeight = shape[2].toInt()
                inputWidth = shape[3].toInt()
            }
        } catch (e: Exception) {
            loadError = e.message ?: "文件读取失败"
        }
    }

    override fun recognize(screen: Bitmap): GameState? {
        val s = session ?: return GameState(Role.LANDLORD, emptyList(), emptyList(), emptyList(), 0, debugInfo = "报错:$loadError")
        
        try {
            // 使用自动探测到的尺寸进行缩放
            val resized = Bitmap.createScaledBitmap(screen, inputWidth, inputHeight, true)
            val inputBuffer = convertBitmapToFloatBuffer(resized)

            val inputTensor = OnnxTensor.createTensor(
                env, inputBuffer,
                longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
            )
            
            val inputName = s.inputNames.iterator().next()
            val inputs = mapOf(inputName to inputTensor)
            val result = s.run(inputs)
            val output = result.get(0) as OnnxTensor
            
            val shape = output.info.shape
            val shapeStr = shape.joinToString(",")
            val outputArray = output.floatBuffer.array()
            result.close()
            inputTensor.close()

            val myHand = mutableListOf<Int>()
            
            if (shape.size != 3) return GameState(Role.LANDLORD, emptyList(), emptyList(), emptyList(), 0, debugInfo = "维度异常:$shapeStr")

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
                 return GameState(Role.LANDLORD, emptyList(), emptyList(), emptyList(), 0, debugInfo = "无法识别维度:$shapeStr")
            }

            val screenHeightThreshold = screen.height * 0.6f 
            var maxConfOverall = 0f

            for (i in 0 until numBoxes) {
                var maxProb = 0f
                var classId = -1
                
                for (c in 0 until numClasses) {
                    val probIndex = if (isChannelFirst) (4 + c) * numBoxes + i else i * (4 + numClasses) + (4 + c)
                    val prob = outputArray[probIndex]
                    if (prob > maxProb) {
                        maxProb = prob
                        classId = c
                    }
                }

                if (maxProb > maxConfOverall) maxConfOverall = maxProb

                if (maxProb > 0.1f && classId != -1) {
                    val cyIndex = if (isChannelFirst) 1 * numBoxes + i else i * (4 + numClasses) + 1
                    val cy = outputArray[cyIndex]
                    val realY = cy * screen.height 

                    if (realY > screenHeightThreshold) {
                        myHand.add(classId)
                    }
                }
            }

            // 加上尺寸提示，方便我们观察
            val debugStr = "尺寸:${inputWidth}x${inputHeight} 最高概率:${String.format("%.2f", maxConfOverall)}"

            if (myHand.isEmpty()) {
                return GameState(Role.LANDLORD, emptyList(), emptyList(), emptyList(), 0, debugInfo = debugStr)
            }

            return GameState(Role.LANDLORD, myHand, listOf(emptyList(), emptyList(), emptyList()), listOf(17, 17), 0, debugInfo = "识别成功")
        } catch (e: Exception) {
            return GameState(Role.LANDLORD, emptyList(), emptyList(), emptyList(), 0, debugInfo = "推理错:${e.message}")
        }
    }

    private fun convertBitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val floatBuffer = FloatBuffer.allocate(3 * width * height)
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val rChannel = FloatArray(width * height)
        val gChannel = FloatArray(width * height)
        val bChannel = FloatArray(width * height)

        for (i in pixels.indices) {
            val p = pixels[i]
            rChannel[i] = ((p shr 16 and 0xFF) / 255.0f)
            gChannel[i] = ((p shr 8 and 0xFF) / 255.0f)
            bChannel[i] = ((p and 0xFF) / 255.0f)
        }
        floatBuffer.put(rChannel)
        floatBuffer.put(gChannel)
        floatBuffer.put(bChannel)
        floatBuffer.rewind()
        return floatBuffer
    }
}

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
