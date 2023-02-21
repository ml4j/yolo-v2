/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ml4j.nn.models.yolov2.impl;

import java.util.Arrays;

import org.ml4j.nn.architectures.yolo.yolov2.YOLOv2WeightsLoader;
import org.ml4j.nn.axons.BiasVector;
import org.ml4j.nn.axons.BiasVectorImpl;
import org.ml4j.nn.axons.FeaturesVector;
import org.ml4j.nn.axons.FeaturesVectorFormatImpl;
import org.ml4j.nn.axons.FeaturesVectorImpl;
import org.ml4j.nn.axons.FeaturesVectorOrientation;
import org.ml4j.nn.axons.WeightsFormatImpl;
import org.ml4j.nn.axons.WeightsMatrix;
import org.ml4j.nn.axons.WeightsMatrixImpl;
import org.ml4j.nn.axons.WeightsMatrixOrientation;
import org.ml4j.nn.axons.WeightsVector;
import org.ml4j.nn.axons.WeightsVectorImpl;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;

/**
 * @author Michael Lavelle
 */
public class DefaultUntrainedYOLOv2WeightsLoader implements YOLOv2WeightsLoader {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public WeightsMatrix getConvolutionalLayerWeights(String name, int width, int height, int inputDepth, int outputDepth) {
		if (width == 1 && height == 1) {
			return new WeightsMatrixImpl(null,
					new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH), 
							Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
		} else {
			return new WeightsMatrixImpl(null,
					new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH), 
							Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
		}
	}

	@Override
	public WeightsVector getBatchNormLayerGamma(String name, int inputDepth) {
		return new WeightsVectorImpl(null,
				new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH),
						FeaturesVectorOrientation.COLUMN_VECTOR, DimensionScope.OUTPUT));
	}

	@Override
	public FeaturesVector getBatchNormLayerMovingVariance(String name, int inputDepth) {
		return new FeaturesVectorImpl(null, new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH),
				FeaturesVectorOrientation.COLUMN_VECTOR, DimensionScope.OUTPUT));
	}

	@Override
	public FeaturesVector getBatchNormLayerMovingMean(String name, int inputDepth) {
		return new FeaturesVectorImpl(null, new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH),
				FeaturesVectorOrientation.COLUMN_VECTOR, DimensionScope.OUTPUT));
	}

	@Override
	public BiasVector getBatchNormLayerBeta(String name, int inputDepth) {
		return new BiasVectorImpl(null, new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH),
				FeaturesVectorOrientation.COLUMN_VECTOR, DimensionScope.OUTPUT));
	}

	@Override
	public BiasVector getConvolutionalLayerBiases(String name, int outputDepth) {
		return new BiasVectorImpl(null, new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH),
				FeaturesVectorOrientation.COLUMN_VECTOR, DimensionScope.OUTPUT));
	}

}
