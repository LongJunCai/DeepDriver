package deepDriver.dl.aml.lstm.apps.util;

import java.util.List;

/**
 * Feature Factory to control the embedding feature
 * 
 * */
public class FeatureFactory {
	
	Embedding embedding;
	int dim; // embedding dimensions
	
	public FeatureFactory(Embedding embedding) {	
		this.embedding = embedding;
		this.dim = embedding.dim;
	}
	
	/**
	 * Concatenate inputs to the first layer of LSTM model
	 * */
	
	public double[] getEmbedFeature(List<TaggedWord> sen, int index, int window) {
		int ngram = 2 * window + 1;
		int length = sen.size(); // length of sentences
		double[][] mat = new double[ngram][dim];
		for (int i = (index - window); i < (index + window + 1); i++) {
			int s = i - (index - window);
			if (i >= 0 && i < length) {
				mat[s] = embedding.getWordVec(sen.get(i).word());
			} else if (i < length) { // 0 padding
				mat[s] = new double[dim];
			}
		}
		return concat(mat);
	}
	
	private double[] concat(double[][] mat){
		int nrow = mat.length;
		int ncol = mat[0].length;
		double[] vec = new double[nrow * ncol];
		for (int i = 0; i < nrow; i++) {
			for (int j = 0; j < ncol; j++) {
				int idx = i * ncol + j;
				vec[idx] = mat[i][j];
			}
		}
		return vec;
	}
	
	public double[] getOneHotFeature(List<TaggedWord> sen, int index) {
		// To Do
		return null;
	}
	
}