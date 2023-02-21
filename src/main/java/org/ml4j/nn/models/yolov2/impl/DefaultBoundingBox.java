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

import org.ml4j.Matrix;
import org.ml4j.nn.models.yolov2.BoundingBox;

/**
 * Details for a YOLOv2 bounding box, with defined corners, 
 * a predicted class index and a predicted class score.
 * 
 * Instantiated with Matrices from the DefaultYOLOv2BoundingBoxExtractor since ported from python code 
 * - TODO refactor.
 * 
 * @author Michael Lavelle
 */
public class DefaultBoundingBox implements BoundingBox {
	
	private Matrix boxXY;
	private Matrix boxWH;
	private Matrix boxClassProbs;
	private float confidence;
	
	public DefaultBoundingBox(Matrix boxXY, Matrix boxWH, Matrix boxClassProbs, float confidence) {
		super();
		this.boxXY = boxXY;
		this.boxWH = boxWH;
		this.boxClassProbs = boxClassProbs;
		this.confidence = confidence;
	}
	
	@Override
	public int getPredictedClassIndex() {
		return getScores().argmax();
	}
	
	@Override
	public float getPredictedClassScore() {
		return getScores().get(getPredictedClassIndex());
	}
	
	private Matrix getScores() {
		return boxClassProbs.mul(confidence);
	}
	
	/**
	 * @return min y, min x, max y, max x,  as pixel coordinates of a originalWidth * originalHeight image
	 */
	@Override
	public float[] getScaledCorners(int originalWidth, int originalHeight) {
		float[] corners = getScaledCorners();
		corners[0] = corners[0] * originalHeight / 608;
		corners[1] = corners[1] * originalWidth / 608;
		corners[2] = corners[2] * originalHeight / 608;
		corners[3] = corners[3] * originalWidth / 608;
		return corners;
	}
	
	/**
	 * @return min y, min x, max y, max x, as pixel coordinates of a 608 * 608 image; 
	 */
	@Override
	public float[] getScaledCorners() {
		float[] result = new float[4];
		int ind = 0;
		for (float f : getCorners()) {
			f = f * 608;
			result[ind] = f;
			ind++;
		}
		return result;
	}
	
	/**
	 * @return min y, min x, max y, max x ( percentages)
	 */
	private float[] getCorners() {
		
		Matrix boxMins = boxXY.sub(boxWH.div(2));
		Matrix boxMaxs = boxXY.add(boxWH.div(2));
		return new float[] {boxMins.get(1), boxMins.get(0), boxMaxs.get(1), boxMaxs.get(0)};
	}
}