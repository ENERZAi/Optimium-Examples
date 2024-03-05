package com.enerzai.optimium.example.android

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.util.LinkedList
import kotlin.math.exp
import kotlin.math.max

internal data class FloatPair(val x: Float, val y: Float)

private const val WIDTH = 128
private const val HEIGHT = 128

private const val MIN_SCORE = 0.5f
private const val RAW_SCORE_LIMIT = 80f
private const val CLASSES_COUNT = 896
private const val BOX_COORD_OFFSET = 16
private const val MIN_SUPPRESSION_THRESHOLD = 0.3f
//private const val MIN_SUPPRESSION_THRESHOLD = 0.1f

internal fun calculateAnchors(): List<FloatPair> {
    // Copied from mediapipe/modules/face_detection/face_detection_short_range.pbtxt
    // and mediapipe/modules/face_detection/face_detection.pbtxt
    val numLayers = 4
    val inputSizeHeight = 128
    val inputSizeWidth = 128
    val anchorOffsetX = 0.5f
    val anchorOffsetY = 0.5f
    val strides = arrayOf(8, 16, 16, 16)
//    val interpolatedScaleAspectRatio = 1f

    var layerId = 0
    val anchors = arrayListOf<FloatPair>()

    while (layerId < numLayers) {
        var lastSameStrideLayer = layerId
        var repeats = 0

        while (lastSameStrideLayer < numLayers
            && strides[lastSameStrideLayer] == strides[layerId]
        ) {
            lastSameStrideLayer += 1

            repeats += 2 // (if (interpolatedScaleAspectRatio == 1.0f) 2 else 1)
        }

        val stride = strides[layerId]
        val featureMapHeight = inputSizeHeight / stride
        val featureMapWidth = inputSizeWidth / stride

        for (y in 0 until featureMapHeight) {
            val yCenter = (y + anchorOffsetY) / featureMapHeight

            for (x in 0 until featureMapWidth) {
                val xCenter = (x + anchorOffsetX) / featureMapWidth

                for (unused in 0 until repeats) {
                    anchors.add(FloatPair(xCenter, yCenter))
                }
            }
        }

        layerId = lastSameStrideLayer
    }

    return anchors
}

private fun decodeBoxes(scale: Float, anchors: List<FloatPair>, data: FloatArray) {
    // shape: 1 * 896 * 16
    // reshape: 1x896x16 -> 896x8x2
    // raw_boxes = raw_boxes.reshape(-1, points, 2)
    // box order: y_center x_center height width

    for (i in 0 until CLASSES_COUNT) {
        val anchor = anchors[i]

        // Ignore key points. Uses detection box only
        val offset = i * BOX_COORD_OFFSET

        var yCenter = data[offset + 0]
        var xCenter = data[offset + 1]
        var h = data[offset + 2]
        var w = data[offset + 3]

        xCenter = xCenter / scale + anchor.x
        yCenter = yCenter / scale + anchor.y

        h /= scale
        w /= scale

        val ymin = yCenter - h / 2.0f
        val xmin = xCenter - w / 2.0f
        val ymax = yCenter + h / 2.0f
        val xmax = xCenter + w / 2.0f

        data[offset + 0] = ymin
        data[offset + 1] = xmin
        data[offset + 2] = ymax
        data[offset + 3] = xmax
    }
}

private fun postprocessScores(data: FloatArray) {
    for (i in data.indices) {
        if (data[i] < -RAW_SCORE_LIMIT)
            data[i] = -RAW_SCORE_LIMIT
        else if (data[i] > RAW_SCORE_LIMIT)
            data[i] = RAW_SCORE_LIMIT

        data[i] = 1.0f / (1.0f + exp(-data[i]))
    }
}

private fun isValid(boxes: FloatArray, offset: Int): Boolean =
    boxes[offset + 2] > boxes[offset + 0] && boxes[offset + 3] > boxes[offset + 1]

private fun convertToDetections(boxes: FloatArray, scores: FloatArray): List<Detection> {
    val detections = arrayListOf<Detection>()

    for (i in 0 until CLASSES_COUNT) {
        if (scores[i] <= MIN_SCORE)
            continue

        val boxOffset = i * BOX_COORD_OFFSET
        if (isValid(boxes, boxOffset)) {
            val ymin = boxes[boxOffset + 0]
            val xmin = boxes[boxOffset + 1]
            val ymax = boxes[boxOffset + 2]
            val xmax = boxes[boxOffset + 3]

            detections.add(Detection(BBox(xmin, ymin, xmax, ymax), scores[i]))
        }
    }

    return detections
}

data class IntFloatPair(val i: Int, val f: Float)

private fun makeStr(data: List<IntFloatPair>): String {
    val buffer = StringBuilder()

    buffer.append("[")

    data.forEach {
        buffer.append("(").append(it.i).append(", ").append(it.f).append("), ")
    }

    buffer.append("]")

    return buffer.toString()
}

internal fun nonMaximumSuppression(detections: List<Detection>): List<Detection> {
    val indexedScores = detections.mapIndexed { index, det -> IntFloatPair(index, det.score) }.sortedByDescending { it.f }
    val outputs = arrayListOf<Detection>()

    var remaining = LinkedList(indexedScores)
    val candidate = LinkedList<IntFloatPair>()
    var backed = LinkedList<IntFloatPair>()

    while (remaining.isNotEmpty()) {
        Log.d(TAG, "remaining = ${makeStr(remaining)}")
        Log.d(TAG, "candidate = ${makeStr(candidate)}")
        Log.d(TAG, "backed = ${makeStr(backed)}")

        val (index, score) = remaining.first()

        if (score < MIN_SCORE)
            break

        val prevScores = remaining.size
        val detection = detections[index]

        backed.clear()
        candidate.clear()

        for (pair in remaining) {
            val box = detections[pair.i].box

            val similarity = box.iou(detection.box)

            if (similarity > MIN_SUPPRESSION_THRESHOLD) {
                Log.d(TAG, "box($box) ~= detection($detection): $similarity")
                candidate.add(pair)
            } else {
                Log.d(TAG, "box($box) != detection($detection): $similarity")
                backed.add(pair)
            }
        }

        val weightedDetection = if (candidate.isEmpty()) {
            detection
        } else {
            var totalScore = 0f
            var weightedXMin = 0f
            var weightedYMin = 0f
            var weightedXMax = 0f
            var weightedYMax = 0f

            candidate.forEach {
                val (index, score) = it

                totalScore += score

                val box = detections[index].box

                weightedXMin += box.xMin * score
                weightedYMin += box.yMin * score
                weightedXMax += box.xMax * score
                weightedYMax += box.yMax * score
            }

            weightedXMin /= totalScore
            weightedYMin /= totalScore
            weightedXMax /= totalScore
            weightedYMax /= totalScore

            Detection(
                BBox(weightedXMin, weightedYMin, weightedXMax, weightedYMax),
                detection.score
            )
        }

        outputs.add(weightedDetection)

        if (prevScores == backed.size)
            break

        run {
            val temp = backed
            backed = remaining
            remaining = temp
        }
    }

    return outputs
}

internal fun removeLetterboxes(
    detections: List<Detection>,
    padding: FloatPair
): List<Detection> {
    val hScale = 1f - (padding.x * 2)
    val vScale = 1f - (padding.y * 2)

    return detections.map {
        val xmin = (it.box.xMin - padding.x) / hScale
        val ymin = (it.box.yMin - padding.y) / vScale
        val xmax = (it.box.xMax - padding.x) / hScale
        val ymax = (it.box.yMax - padding.y) / vScale

        Detection(BBox(xmin, ymin, xmax, ymax), it.score)
    }
}

internal fun resizeData(input: Bitmap, rotated: Boolean, maxSize: Int = 1024): Mat {
    val inputROI = Rect(input.width, input.height)

    val size = max(input.width, input.height)
    val scale = if (size > maxSize) { maxSize.toFloat() / size.toFloat() } else { 1f }

    val inputMat = Mat()
    val outputMat = Mat()

    Utils.bitmapToMat(input, inputMat)
    Imgproc.resize(inputMat, outputMat, inputROI.scaled(scale, scale).toSize())

    if (rotated) {
        Core.rotate(outputMat, outputMat, Core.ROTATE_90_CLOCKWISE)
    }

    return outputMat
}

internal fun preprocessData(data: Mat, buffer: FloatArray): FloatPair {
    val inputROI = Rect(data.width(), data.height())
    val outputROI = if (inputROI.aspectRatio > 1.0f) {
        Rect.IDENTITY.scaled(WIDTH / inputROI.aspectRatio, HEIGHT.toFloat())
    } else {
        Rect.IDENTITY.scaled(WIDTH.toFloat(), HEIGHT * inputROI.aspectRatio)
    }

    val resized = Mat()
    val widthPadding = ((WIDTH - outputROI.width) / 2)
    val heightPadding = ((HEIGHT - outputROI.height) / 2)

    val left = widthPadding
    val right = widthPadding + (outputROI.width % 2)
    val top = heightPadding
    val bottom = heightPadding + (outputROI.height % 2)

    // Resize
    Imgproc.resize(data, resized, outputROI.toSize())

    val padded = Mat()
    Core.copyMakeBorder(resized, padded, top, bottom, left, right, Core.BORDER_CONSTANT, Scalar(0.0))

    // Convert
    val converted = Mat(WIDTH, HEIGHT, CvType.CV_8UC3)

    // RGBA to RGB
    Core.mixChannels(listOf(padded), listOf(converted), MatOfInt(0, 0, 1, 1, 2, 2))

    // u8 to f32, normalize [0, 255] to [0, 1)
    val result = Mat(WIDTH, HEIGHT, CvType.CV_32FC3)
    converted.convertTo(result, CvType.CV_32FC3, 1.0 / 255.0)

    // Copy
    result.get(intArrayOf(0, 0), buffer)

    resized.release()
    converted.release()
    result.release()

    return FloatPair(
        widthPadding.toFloat() / WIDTH.toFloat(),
        heightPadding.toFloat() / HEIGHT.toFloat()
    )
}

internal fun postprocessData(regressors: FloatArray, classficators: FloatArray, anchors: List<FloatPair>, roi: FloatPair): List<Detection> {
    decodeBoxes(WIDTH.toFloat(), anchors, regressors)
    postprocessScores(classficators)

    val detections = convertToDetections(regressors, classficators)
    val suppressed = nonMaximumSuppression(detections)
    return removeLetterboxes(suppressed, roi)
}

internal fun drawDetections(image: Mat, results: List<Detection>, output: Bitmap, color: Scalar, thickness: Int) {
    results.forEach {
        val rect = it.box.scale(image.width(), image.height()).toRect()
        Imgproc.rectangle(image, rect, color, thickness)
    }

    Utils.matToBitmap(image, output)
}