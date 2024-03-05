package com.enerzai.optimium.example.android

import org.opencv.core.Rect
import kotlin.math.max
import kotlin.math.min

data class BBox(val xMin: Float, val yMin: Float, val xMax: Float, val yMax: Float) {
    val width: Float = xMax - xMin

    val height: Float = yMax - yMin

    val empty: Boolean = width <= 0 || height <= 0

    val area: Float = if (!empty) { width * height } else { 0.0f }

    fun scale(scaleX: Float, scaleY: Float): BBox =
        BBox(xMin * scaleX, yMin * scaleY, xMax * scaleX, yMax * scaleY)

    fun scale(scaleX: Int, scaleY: Int): BBox = scale(scaleX.toFloat(), scaleY.toFloat())

    fun intersect(other: BBox): BBox? {
        val xmin = max(xMin, other.xMin)
        val ymin = max(yMin, other.yMin)
        val xmax = min(xMax, other.xMax)
        val ymax = min(yMax, other.yMax)

        return if (xmin < xmax && ymin < ymax) {
            BBox(xmin, ymin, xmax, ymax)
        } else {
            null
        }
    }

    fun iou(other: BBox): Float {
        val intersection = intersect(other) ?: return 0f
        val denominator = this.area + other.area - intersection.area
        return if (denominator > 0f) {
            intersection.area / denominator
        } else {
            0f
        }
    }

    fun toRect(): Rect = Rect(
        xMin.toInt(), yMin.toInt(), width.toInt(), height.toInt()
    )
}