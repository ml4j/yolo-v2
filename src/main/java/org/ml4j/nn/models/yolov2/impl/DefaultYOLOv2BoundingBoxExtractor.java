/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.models.yolov2.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.models.yolov2.BoundingBox;
import org.ml4j.nn.models.yolov2.BoundingBoxExtractor;
import org.ml4j.nn.models.yolov2.YOLOv2Labels;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationContextImpl;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

/**
 * Default implementation of BoundingBoxExtractor, responsible for obtaining a list of BoundingBoxes from the float[] output of a YOLO network for a single image example, 
 * given a BoundingBox score threshold and an iouThreshold for non max suppression.
 * 
 * @author Michael Lavelle
 *
 */
public class DefaultYOLOv2BoundingBoxExtractor implements BoundingBoxExtractor {

	private DifferentiableActivationFunction softmaxActivationFunction;
	private MatrixFactory matrixFactory;
	private float[][] anchors;

	public DefaultYOLOv2BoundingBoxExtractor(MatrixFactory matrixFactory,
			DifferentiableActivationFunction softmaxActivationFunction) {
		this.matrixFactory = matrixFactory;
		this.softmaxActivationFunction = softmaxActivationFunction;
		this.anchors = new float[5][2];
		anchors[0] = new float[] { 0.57273f, 0.677385f };
		anchors[1] = new float[] { 1.87446f, 2.06253f };
		anchors[2] = new float[] { 3.33843f, 5.47434f };
		anchors[3] = new float[] { 7.88282f, 3.52778f };
		anchors[4] = new float[] { 9.77052f, 9.16828f };
	}
	
	public List<DefaultBoundingBox> getScoreFilteredBoundingBoxes(float[] data, YOLOv2Labels yoloV2ClassificationNames, float scoreThreshold) {
		return getAllBoundingBoxes(data, yoloV2ClassificationNames).stream().filter(b -> b.getPredictedClassScore() > scoreThreshold).collect(Collectors.toList());
	}
	
	public List<BoundingBox> getScoreFilteredBoundingBoxesWithNonMaxSuppression(float[] data, YOLOv2Labels yoloV2ClassificationNames, 
			float scoreThreshold, float iouThreshold) {
		return applyNonMaxSuppression(getAllBoundingBoxes(data, yoloV2ClassificationNames).stream().filter(
				b -> b.getPredictedClassScore() > scoreThreshold).collect(Collectors.toList()),iouThreshold);
	}

	/**
	 * Obtains the bounding boxes given the raw YOLO output data ( float[] of 425 * 19 * 19) for a single image).
	 * 
	 * TODO - tidy up and vectorise this method to operate on the data for multiple images at once
	 * @param data
	 * @param yoloV2ClassificationNames
	 * @return A list of bounding boxes
	 */
	private List<DefaultBoundingBox> getAllBoundingBoxes(float[] data, YOLOv2Labels yoloV2ClassificationNames) {

		List<DefaultBoundingBox> results = new ArrayList<>();

		int index = 0;
		// TODO -externalise these anchor definitions.
		float[][] anchors = new float[5][2];
		anchors[0] = new float[] { 0.57273f, 0.677385f };
		anchors[1] = new float[] { 1.87446f, 2.06253f };
		anchors[2] = new float[] { 3.33843f, 5.47434f };
		anchors[3] = new float[] { 7.88282f, 3.52778f };
		anchors[4] = new float[] { 9.77052f, 9.16828f };

		for (int r = 0; r < 19; r++) {
			for (int c = 0; c < 19; c++) {
				for (int b = 0; b < 5; b++) {
					float[] bData = new float[5];
					float[] cData = new float[80];
					System.arraycopy(data, index, bData, 0, 5);
					System.arraycopy(data, index + 5, cData, 0, 80);

					Matrix classProbs = matrixFactory.createMatrixFromRowsByRowsArray(80, 1, cData);
					Matrix boxConfidence = matrixFactory.createMatrixFromRowsByRowsArray(1, 1, new float[] { bData[4] })
							.sigmoid();

					Matrix boxXY = matrixFactory
							.createMatrixFromRowsByRowsArray(2, 1, new float[] { bData[0], bData[1] }).sigmoid();
					Matrix anchorsMatrix = matrixFactory.createMatrixFromRowsByRowsArray(2, 1, anchors[b]);
					Matrix boxWH = matrixFactory
							.createMatrixFromRowsByRowsArray(2, 1, new float[] { bData[2], bData[3] })
							.asEditableMatrix().expi();

					Matrix convDims = matrixFactory.createMatrixFromRowsByRowsArray(2, 1, new float[] { 19, 19 });

					Matrix convIndex = matrixFactory.createMatrixFromRowsByRowsArray(2, 1, new float[] { c, r });

					boxXY = boxXY.add(convIndex).div(convDims);
					boxWH = boxWH.mul(anchorsMatrix).div(convDims);

					float confidence = boxConfidence.getRowByRowArray()[0];

					NeuronsActivationContext context = new NeuronsActivationContextImpl(matrixFactory, false);
					DifferentiableActivationFunctionActivation act = softmaxActivationFunction
							.activate(new NeuronsActivationImpl(new Neurons(80, false), classProbs,
									NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET), context);

					Matrix box_class_probs = act.getOutput().getActivations(matrixFactory);
					DefaultBoundingBox result = new DefaultBoundingBox(boxXY, boxWH, box_class_probs, confidence);
					results.add(result);
					index = index + 85;
				}

			}
		}
		return results;
	}

	private float getIOU(BoundingBox first, BoundingBox second) {

		float xi1 = (float) Math.max(first.getScaledCorners()[0], second.getScaledCorners()[0]);
		float yi1 = (float) Math.max(first.getScaledCorners()[1], second.getScaledCorners()[1]);
		float xi2 = (float) Math.min(first.getScaledCorners()[2], second.getScaledCorners()[2]);
		float yi2 = (float) Math.min(first.getScaledCorners()[3], second.getScaledCorners()[3]);

		float inter_area = (yi2 - yi1) * (xi2 - xi1);

		float box1_area = ((first.getScaledCorners()[3] - first.getScaledCorners()[1])
				* (first.getScaledCorners()[2] - first.getScaledCorners()[0]));
		float box2_area = ((second.getScaledCorners()[3] - second.getScaledCorners()[1])
				* (second.getScaledCorners()[2] - second.getScaledCorners()[0]));

		float union_area = box1_area + box2_area - inter_area;

		float iou = inter_area / union_area;

		return iou;

	}

	public List<BoundingBox> applyNonMaxSuppression(List<BoundingBox> b, float iouThreshold) {

		List<BoundingBox> d = new ArrayList<>();
		while (!b.isEmpty()) {
			BoundingBox max = getMax(b);
			d.add(max);
			b.remove(max);
			List<BoundingBox> toRemove = new ArrayList<>();
			for (BoundingBox r : b) {
				float iou = getIOU(r, max);
				if (iou > iouThreshold) {
					toRemove.add(r);
				}
			}
			for (BoundingBox remove : toRemove) {
				b.remove(remove);
			}
		}
		return d;
	}

	private BoundingBox getMax(List<BoundingBox> results) {
		BoundingBox max = null;
		Float maxScore = null;
		for (BoundingBox r : results) {
			if (maxScore == null || r.getPredictedClassScore() > maxScore.floatValue()) {
				maxScore = r.getPredictedClassScore();
				max = r;
			}
		}
		return max;
	}
}
