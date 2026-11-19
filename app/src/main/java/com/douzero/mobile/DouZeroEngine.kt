package com.douzero.mobile

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntimepackage com.douzero.mobile

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

class MockCardRecognizer : CardRecognizer {
    override fun recognize(screen: Bitmap): GameState? {
        val hand = mutableListOf<Int>()
        repeat(2) { hand.add(0) }
        for (v in 1..12) hand.add(v)
        hand.add(13)
        return GameState(
            myRole = Role.LANDLORD,
            myHand = hand,
            lastMoves = listOf(emptyList(), listOf(1), emptyList()),
            otherCardCounts = listOf(10, 12),
            bombCount = 0
        )
    }
}

class DouZeroEngine(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val sessions = mutableMapOf<Role, OrtSession>()

    init {
        listOf(
            Role.LANDLORD to "landlord.onnx",
            Role.LANDLORD_UP to "landlord_up.onnx",
            Role.LANDLORD_DOWN to "landlord_down.onnx"
        ).forEach { (role, name) ->
            try {
                val f = File(context.filesDir, name)
                if (f.exists()) {
                    sessions[role] = env.createSession(f.absolutePath, OrtSession.SessionOptions())
                }
            } catch (_: Exception) { }
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

    // 这里修复了报错的代码，改成了直接生成一维数组
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
}.OrtSession
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

class MockCardRecognizer : CardRecognizer {
    override fun recognize(screen: Bitmap): GameState? {
        val hand = mutableListOf<Int>()
        repeat(2) { hand.add(0) }
        for (v in 1..12) hand.add(v)
        hand.add(13)
        return GameState(
            myRole = Role.LANDLORD,
            myHand = hand,
            lastMoves = listOf(emptyList(), listOf(1), emptyList()),
            otherCardCounts = listOf(10, 12),
            bombCount = 0
        )
    }
}

class DouZeroEngine(context: Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val sessions = mutableMapOf<Role, OrtSession>()

    init {
        listOf(
            Role.LANDLORD to "landlord.onnx",
            Role.LANDLORD_UP to "landlord_up.onnx",
            Role.LANDLORD_DOWN to "landlord_down.onnx"
        ).forEach { (role, name) ->
            try {
                val f = File(context.filesDir, name)
                if (f.exists()) {
                    sessions[role] = env.createSession(f.absolutePath, OrtSession.SessionOptions())
                }
            } catch (_: Exception) { }
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
        val mat = Array(4) { FloatArray(15) }
        val counts = IntArray(15)
        hand.forEach { if (it in 0..14) counts[it]++ }
        for (v in 0 until 15) {
            for (r in 0 until counts[v].coerceAtMost(4)) mat[r][v] = 1f
        }
        return mat.flatten().toFloatArray()
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
