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
package org.ml4j.nn.models.yolov2;

import java.util.List;

/**
 * Responsible for obtaining a list of BoundingBoxes from the float[] output of a YOLO network for a single image example, 
 * given a BoundingBox score threshold and an iouThreshold for non max suppression.
 * 
 * @author Michael Lavelle
 *
 */
public interface BoundingBoxExtractor {

	/**
	 * @param data The output for a single image example from the YOLO network - a tensor of shape 425 * 19 * 19.
	 * @param yoloV2ClassificationNames The classification names of the YOLO network.
	 * @param scoreThreshold The BoundingBox score threshold.
	 * @param iouThreshold The iou threshold.
	 * @return A list of score-filtered BoundingBox instances, with non max suppression applied.
	 */
	List<BoundingBox> getScoreFilteredBoundingBoxesWithNonMaxSuppression(float[] data, YOLOv2Labels yoloV2ClassificationNames, 
			float scoreThreshold, float iouThreshold);
}
