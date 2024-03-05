package com.enerzai.optimium.example.android

import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size

data class Rect(val width: Int, val height: Int) {
    companion object {
        val IDENTITY = Rect(1, 1)
    }

    fun scaled(scaleX: Float, scaleY: Float): Rect =
        Rect((width * scaleX).toInt(), (height * scaleY).toInt())

    val aspectRatio: Float = height.toFloat() / width.toFloat()

    fun toSize(): Size =
        Size(width.toDouble(), height.toDouble())

    fun points(): Mat = MatOfPoint2f(
        Point(0.0, 0.0),
        Point(width.toDouble(), 0.0),
        Point(width.toDouble(), height.toDouble()),
        Point(0.0, height.toDouble())
    )
}