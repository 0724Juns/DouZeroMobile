package com.douzero.mobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
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
    val isDoublingPhase: Boolean = false
)

interface CardRecognizer {
    fun recognize(screen: Bitmap): GameState?
}

// ================= 视觉识别模块 =================
class YoloCardRecognizer(context: Context) : CardRecognizer {
    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val inputSize = 960 // yolov11n960 的输入尺寸

    // 类别映射：0-12 是 3到2，13是小王，14是大王
    // 假设模型训练时的类别顺序就是标准的牌面顺序

    init {
        try {
            // 注意：这里必须换成你上传的 .onnx 文件准确名字！
            val modelName = "best.onnx" 
            val f = File(context.filesDir, modelName)
            if (!f.exists()) {
                context.assets.open(modelName).use { input ->
                    f.outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
            session = env.createSession(f.absolutePath, OrtSession.SessionOptions())
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    override fun recognize(screen: Bitmap): GameState? {
        val s = session ?: return null

        try {
            // 1. 图像预处理：缩放到 960x960，并转为 FloatBuffer
            val resized = Bitmap.createScaledBitmap(screen, inputSize, inputSize, true)
            val inputBuffer = convertBitmapToFloatBuffer(resized)

            // 2. 构建输入张量 (1, 3, 960, 960)
            val inputTensor = OnnxTensor.createTensor(
                env, inputBuffer,
                longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
            )
            
            // 注意：YOLOv11 的输入名字通常是 "images" 或者 "input"
            val inputs = mapOf("images" to inputTensor)
            val result = s.run(inputs)
            val output = result.get(0) as OnnxTensor
            val outputArray = output.floatBuffer.array()
            result.close()
            inputTensor.close()

            // 3. 解析 YOLO 输出并提取手牌 (这是一个简化版的解析逻辑，真实场景下需要 NMS)
            // YOLOv11 的输出通常是 [1, 4+15, 18900] (cx, cy, w, h + 15个类别的概率)
            // 这里的解析非常复杂，如果解析不对，手牌可能就是空的。
            
            // 【重点】由于没有现成的 NMS (非极大值抑制) 代码，我们用个取巧的办法：
            // 我们只统计屏幕下方 30% 区域内的检测框（假设为手牌）
            val myHand = mutableListOf<Int>()
            val screenHeightThreshold = screen.height * 0.7f // 屏幕下方 30%

            // 假设输出格式是 [1, 19, 18900] (YOLOv11 常见格式)
            val numClasses = 15
            val numBoxes = 18900
            val elementsPerBox = 4 + numClasses

            for (i in 0 until numBoxes) {
                // 找出这个框的 15 个类别的最大概率和对应类别
                var maxProb = 0f
                var classId = -1
                for (c in 0 until numClasses) {
                    val prob = outputArray[(4 + c) * numBoxes + i]
                    if (prob > maxProb) {
                        maxProb = prob
                        classId = c
                    }
                }

                // 如果概率大于 0.5，且类别有效
                if (maxProb > 0.5f && classId != -1) {
                    // 获取框的 Y 坐标 (cy)
                    val cy = outputArray[1 * numBoxes + i]
                    // 将 Y 坐标还原到屏幕分辨率
                    val realY = cy * screen.height 

                    // 如果在屏幕下方区域，算作手牌
                    if (realY > screenHeightThreshold) {
                        myHand.add(classId)
                    }
                }
            }

            if (myHand.isEmpty()) return null

            // 4. 组装 GameState（暂时忽略上家出牌和叫地主按钮，只让 AI 算手牌）
            return GameState(
                myRole = Role.LANDLORD,
                myHand = myHand,
                lastMoves = listOf(emptyList(), emptyList(), emptyList()), // 暂不识别出牌
                otherCardCounts = listOf(17, 17), // 假的剩余牌数
                bombCount = 0
            )

        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }
    }

    // 将 Bitmap 转换为 FloatBuffer (YOLO 预处理)
    private fun convertBitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val floatBuffer = FloatBuffer.allocate(3 * width * height)

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // YOLO 的输入通常需要归一化到 0-1，并且按照 R, G, B 通道排列 (NCHW)
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

// ================= 算牌模型模块 =================
class DouZeroEngine(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val sessions = mutableMapOf<Role, OrtSession>()

    init {
        listOf(
            Role.LANDLORD to "landlord.onnx",
            Role.LANDLORD_UP to "landler_up.onnx",     
            Role.LANDLORD_DOWN to "landler_down.onnx"  
        ).forEach { (role, name) ->
            try {
                val f = File(context.filesDir, name)
                if (!f.exists()) {
                    context.assets.open(name).use { input ->
                        f.outputStream().use { output ->
                            input.copyTo(output)
                        }
                    }
                }
                sessions[role] = env.createSession(f.absolutePath, OrtSession.SessionOptions())
            } catch (e: Exception) {
                e.printStackTrace() 
            }
        }
    }

    fun hasModel(): Boolean = sessions.isNotEmpty()

    fun infer(state: GameState): FloatArray {
        val session = sessions[state.myRole] ?: return FloatArray(0)
        return try {
            val handTensor = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(encodeHand(state.myHand)),
                longArrayOf(1, 4, 15)
            )
            val inputs = mapOf("hand_cards" to handTensor)
            session.run(inputs).use { result ->
                (result.get(0) as OnnxTensor).floatBuffer.array()
            }
        } catch (_: Exception) {
            FloatArray(0)
        }
    }

    private fun encodeHand(hand: List<Int>): FloatArray {
        val flatArray = FloatArray(4 * 15) 
        val counts = IntArray(15)
        hand.forEach { if (it in 0..14) counts[it]++ }
        for (v in 0 until 15) {
            for (r in 0 until counts[v].coerceAtMost(4)) {
                flatArray[r * 15 + v] = 1f
            }
        }
        return flatArray
    }

    fun close() {
        sessions.values.forEach { it.close() }
        env.close()
    }
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
        return when {
            s >= 8 -> Suggestion("抢地主", emptyList())
            s >= 5 -> Suggestion("考虑抢地主", emptyList())
            else -> Suggestion("不抢", emptyList())
        }
    }

    private fun decideDouble(hand: List<Int>): Suggestion {
        return if (handStrength(hand) >= 7) Suggestion("加倍", emptyList())
        else Suggestion("不加倍", emptyList())
    }

    private fun handStrength(hand: List<Int>): Int {
        var score = 0
        val counts = IntArray(15)
        hand.forEach { if (it in 0..14) counts[it]++ }
        if (counts[13] > 0) score += 3
        if (counts[14] > 0) score += 4
        score += counts[12] * 2
        for (v in 0..12) if (counts[v] == 4) score += 5
        return score
    }

    private fun decidePlay(state: GameState): Suggestion {
        val hand = state.myHand.sorted()
        if (hand.isEmpty()) return Suggestion("无牌可出", emptyList())
        val lastMove = state.lastMoves.lastOrNull { it.isNotEmpty() }
        if (lastMove == null) {
            return Suggestion("出 ${cardName(hand.first())}", listOf(hand.first()))
        }
        if (lastMove.size == 1) {
            val beat = hand.firstOrNull { it > lastMove[0] }
            if (beat != null) return Suggestion("出 ${cardName(beat)}", listOf(beat))
        }
        return Suggestion("过牌", emptyList())
    }

    private fun cardName(v: Int): String = when (v) {
        12 -> "2"; 13 -> "小王"; 14 -> "大王"
        else -> listOf("3","4","5","6","7","8","9","10","J","Q","K")[v]
    }
}
