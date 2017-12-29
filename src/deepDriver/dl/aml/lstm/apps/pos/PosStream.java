package deepDriver.dl.aml.lstm.apps.pos;

import java.util.List;
import java.util.Random;

import deepDriver.dl.aml.lstm.IStream;
import deepDriver.dl.aml.lstm.apps.util.Embedding;
import deepDriver.dl.aml.lstm.apps.util.FeatureFactory;
import deepDriver.dl.aml.lstm.apps.util.TaggedWord;

public class PosStream implements IStream{

	PosDataLoader loader;
	Embedding embedding;
	FeatureFactory factory;	
	
	int tagNum; // Pos Tag Number
	int dim; // embedding dimension
	int window = 1; // window of n-gram 2*window+1
	int featureNum;
	int vSize; // vocabulary size
	int maxLength;
	String UNKNOWN_W = "*";
	String UNKNOWN_T = "NN";
	
	public PosStream(){
		super();
	}
	
	/**
	 * word embedding input
	 * */
	
	public PosStream(PosDataLoader loader, Embedding embedding){
		super();
		this.loader = loader;
		this.embedding = embedding;
		this.factory = new FeatureFactory(embedding);
		this.tagNum = loader.getTagMap().size();
		this.dim = embedding.dim;
		this.featureNum = dim * (2*window + 1);
		this.vSize = embedding.vSize;
		this.maxLength = loader.maxLength;
	}
	
	/**
	 * word one hot vector input
	 * */
	
	public PosStream(PosDataLoader loader){
		super();
		this.loader = loader;
		this.embedding = null;
		this.factory = null;
		this.tagNum = loader.getTagMap().size();
		this.dim = 0; // one hot encoding
		this.featureNum = loader.wordNum;
		this.vSize = loader.wordNum;
		this.maxLength = loader.maxLength;
	}
	
	double[][] sampleTT;
	double[][] targetTT;
	
	int cnt = 0;
	
	@Override
	public void reset() {
		cnt = 0;
	}

	@Override
	public boolean hasNext() {
		return cnt < loader.getSentences().size();
	}

	Random rd = new Random(System.currentTimeMillis());
	@Override
	public void next() {
		cnt ++;
		double ss = loader.getSentences().size();
		int ri = (int)(rd.nextDouble() * ss);
		// int ri = 27483;
		// System.out.println("current example id" + ri);
		
//		List<TaggedWord> sen = loader.getSentences().get(ri);
//		while(sen.size() >= maxLength) {
//			// System.out.println("Sentence + "+ ri + " skiped");
//			ri = (int)(rd.nextDouble() * ss);
//			sen = loader.getSentences().get(ri);
//		}
		next(ri);
	}
	
	@Override
	public double[][] getSampleTT() {
		return sampleTT;
	}

	@Override
	public double[][] getTarget() {
		return targetTT;
	}

	@Override
	public int getSampleTTLength() {
		return sampleTT.length;
	}

	@Override
	public int getSampleFeatureNum() {
		return featureNum;
		// if use word2vec embedding set to nFeature;
		// else use one hot set to vSize
		// return vSize;
	}
	
	@Override
	public int getTargetFeatureNum() {
		return tagNum;
	}
	
	int pos;
	@Override
	public Object getPos() {
		return pos;
	}

	@Override
	public void next(Object pos) {
		this.pos = (Integer) pos; // position
		List<TaggedWord> sen = loader.getSentences().get(this.pos);
		
		int size = sen.size();
		this.sampleTT = new double[size][];
		this.targetTT = new double[size][];
		for (int i = 0; i < size; i++) {
			String word = sen.get(i).word();
			String tag = sen.get(i).tag();
			if (embedding!=null) { //Embedding word2vec input
				sampleTT[i] = factory.getEmbedFeature(sen, i, window); //n-gram
			} else {// One hot input
				sampleTT[i] = new double[1];
				Integer idx = loader.getStrMap().get(word);
				if (idx == null) { // Uknown Character
					idx = loader.getStrMap().get(UNKNOWN_W);
				}
				sampleTT[i][0] = Double.valueOf(idx);				
			}
			
			targetTT[i] = new double[tagNum];
			Integer tagIdx = loader.getTagMap().get(tag);
			if (tagIdx == null) { // Unknown Tags
				if (loader.getTagMap().containsKey("NN")) {
					tagIdx = loader.getTagMap().get("NN");
				}
				if (loader.getTagMap().containsKey("n")) {
					tagIdx = loader.getTagMap().get("n");
				}
				if (tagIdx == null) {
					tagIdx = 0; //default unknown tag
				}
			}
			targetTT[i][tagIdx] = 1;
		}
		
	}
	
	public int getTagNum() {
		return tagNum;
	}

	public void setTagNum(int tagNum) {
		this.tagNum = tagNum;
	}

	@Override
	public IStream[] splitStream(int cnt) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int splitCnt(int cnt) {
		// TODO Auto-generated method stub
		return 0;
	}

}
