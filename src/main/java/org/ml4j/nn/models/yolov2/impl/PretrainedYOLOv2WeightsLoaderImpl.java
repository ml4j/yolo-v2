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

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectStreamClass;
import java.io.Serializable;
import java.util.Arrays;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrix;
import org.ml4j.jblas.JBlasRowMajorMatrixFactory;
import org.ml4j.nn.architectures.yolo.yolov2.YOLOv2WeightsLoader;
import org.ml4j.nn.axons.*;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Michael Lavelle
 */
public class PretrainedYOLOv2WeightsLoaderImpl implements YOLOv2WeightsLoader {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(PretrainedYOLOv2WeightsLoaderImpl.class);

	private MatrixFactory finalMatrixFactory;

	private MatrixFactory matrixFactoryForLoad;
	private ClassLoader classLoader;
	private long uid;

	public PretrainedYOLOv2WeightsLoaderImpl(ClassLoader classLoader, MatrixFactory matrixFactory) {
		this.uid = ObjectStreamClass.lookup(float[].class).getSerialVersionUID();
		this.classLoader = classLoader;
		this.finalMatrixFactory = matrixFactory;
		// Use JBlasRowMajorMatrixFactory for loading weights - as there are (as of yet) unknown
		// performance issues with the methods used from Nd4J Matrix
		this.matrixFactoryForLoad = new JBlasRowMajorMatrixFactory();
	}

	public static PretrainedYOLOv2WeightsLoaderImpl getLoader(MatrixFactory finalMatrixFactory,
			ClassLoader classLoader) {
		return new PretrainedYOLOv2WeightsLoaderImpl(classLoader, finalMatrixFactory);
	}

	private float[] deserializeWeights(String name) {
		LOGGER.debug("Derializing weights:" + name);
		try {
			return deserialize(float[].class, "yolov2javaweights", uid, name);
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
	}

	public WeightsMatrix getConvolutionalLayerWeights(String name, int width, int height, int inputDepth, int outputDepth) {
		float[] weights = deserializeWeights(name);
		// is height, width, inputDepth * outputDepth
		// want outputDepth * inputDepth, height, width.
		Matrix weightsMatrix1 =  matrixFactoryForLoad.createMatrixFromRowsByRowsArray(outputDepth, width * height * inputDepth, weights);

		
		// This is outputDepth * height, width, inputDepth
		Matrix weightsMatrix =  matrixFactoryForLoad.createMatrixFromRowsByRowsArray(width * height * inputDepth, outputDepth, weights).transpose();
		Matrix outputWeights = matrixFactoryForLoad.createMatrix(weightsMatrix.getRows(), weightsMatrix.getColumns());
		for (int r = 0; r < weightsMatrix.getRows(); r++) {
			
			// height, width, inputdepth
			Matrix rowData = weightsMatrix.getRow(r);
			rowData.asEditableMatrix().reshape(inputDepth, height * width);
			Matrix rowData2 = matrixFactoryForLoad.createMatrix(rowData.getRows(), rowData.getColumns());
			for (int i = 0; i < rowData.getRows(); i++) {
				Matrix d = rowData.getRow(i);
				d.asEditableMatrix().reshape(height, width);
				d = d.transpose();
				rowData2.asEditableMatrix().putRow(i, d);
			}


			outputWeights.asEditableMatrix().putRow(r, rowData2);
			
		}
		boolean oneByOneConvolution = width == 1 && height == 1;
		if (oneByOneConvolution) {
			return new WeightsMatrixImpl(finalMatrixFactory.createMatrixFromRowsByRowsArray(weightsMatrix1.getRows(), weightsMatrix1.getColumns(), weightsMatrix1.getRowByRowArray()),
					new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH), 
							Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
		} else {
			return new WeightsMatrixImpl(finalMatrixFactory.createMatrixFromRowsByRowsArray(weightsMatrix1.getRows(), weightsMatrix1.getColumns(), weightsMatrix1.getRowByRowArray()),
					new WeightsFormatImpl(Arrays.asList(Dimension.INPUT_DEPTH, Dimension.FILTER_HEIGHT, Dimension.FILTER_WIDTH), 
							Arrays.asList(Dimension.OUTPUT_DEPTH), WeightsMatrixOrientation.ROWS_SPAN_OUTPUT_DIMENSIONS));
		}
	}

	public WeightsVector getBatchNormLayerGamma(String name, int outputDepth) {
		float[] weights = deserializeWeights(name);
		return new WeightsVectorImpl(finalMatrixFactory.createMatrixFromRowsByRowsArray(outputDepth, 1, weights),
				new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH), FeaturesVectorOrientation.COLUMN_VECTOR,
						DimensionScope.OUTPUT));
	}

	@SuppressWarnings("unchecked")
	public <S extends Serializable> S deserialize(Class<S> clazz, String path, long uid, String id)
			throws IOException, ClassNotFoundException {

		if (classLoader == null) {
			try (InputStream is = new FileInputStream(path + "/" + clazz.getName() + "/" + uid + "/" + id + ".ser")) {
				try (ObjectInputStream ois = new ObjectInputStream(is)) {
					return (S) ois.readObject();
				}
			}
		} else {
			try (InputStream is = classLoader
					.getResourceAsStream(path + "/" + clazz.getName() + "/" + uid + "/" + id + ".ser")) {
				try (ObjectInputStream ois = new ObjectInputStream(is)) {
					return (S) ois.readObject();
				}
			}

		}
	}

	@Override
	public FeaturesVector getBatchNormLayerMovingVariance(String name, int outputDepth) {
		float[] weights = deserializeWeights(name);
		return new FeaturesVectorImpl( finalMatrixFactory.createMatrixFromRowsByRowsArray(outputDepth, 1, weights),
				new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH), FeaturesVectorOrientation.COLUMN_VECTOR,
						DimensionScope.OUTPUT));
	}

	@Override
	public FeaturesVector getBatchNormLayerMovingMean(String name, int outputDepth) {
		float[] weights = deserializeWeights(name);
		return new FeaturesVectorImpl( finalMatrixFactory.createMatrixFromRowsByRowsArray(outputDepth, 1, weights),
				new FeaturesVectorFormatImpl(Arrays.asList(Dimension.OUTPUT_DEPTH), FeaturesVectorOrientation.COLUMN_VECTOR,
						DimensionScope.OUTPUT));
	}

	@Override
	public BiasVector getBatchNormLayerBeta(String name, int outputDepth) {
		float[] weights = deserializeWeights(name);
		return new BiasVectorImpl(finalMatrixFactory.createMatrixFromRowsByRowsArray(outputDepth, 1, weights),
				new BiasFormatImpl(Dimension.OUTPUT_DEPTH, FeaturesVectorOrientation.COLUMN_VECTOR));
	}

	@Override
	public BiasVector getConvolutionalLayerBiases(String name, int outputDepth) {
		float[] weights = deserializeWeights(name);
		return new BiasVectorImpl(finalMatrixFactory.createMatrixFromRowsByRowsArray(outputDepth, 1, weights),
				new BiasFormatImpl(Dimension.OUTPUT_DEPTH, FeaturesVectorOrientation.COLUMN_VECTOR));
	}
}
