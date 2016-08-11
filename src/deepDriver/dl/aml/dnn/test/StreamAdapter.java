package deepDriver.dl.aml.dnn.test;

import java.util.ArrayList;
import java.util.List;


import deepDriver.dl.aml.ann.InputParameters;
import deepDriver.dl.aml.cnn.IDataMatrix;
import deepDriver.dl.aml.cnn.IDataStream;

public class StreamAdapter {
	
	public void loadFromStream(IDataStream is, InputParameters ip) {
		List<double []> inputList = new ArrayList<double[]>();
		List<double []> resultList = new ArrayList<double []>();
		while (is.hasNext()) {
			IDataMatrix dm = is.next(); 
			inputList.add(matrix2Vector(dm.getMatrix()));
			resultList.add(dm.getTarget());
		}
		double [][] in = new double[inputList.size()][];
		double [][] ta = new double[inputList.size()][];
		for (int i = 0; i < ta.length; i++) {
			in[i] = inputList.get(i);
			ta[i] = resultList.get(i);
		}
		ip.setInput(in);
		ip.setResult2(ta);
		inputList.clear();
		resultList.clear();
	}
	
	public double [] matrix2Vector(double [][] matrix) {
		double [] v = new double[matrix.length * matrix[0].length];
		int cnt = 0;
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				v[cnt ++] = matrix[i][j];
			}
		}
		return v;
	}

}
