package deepDriver.dl.aml.lstm;

public class BiRNNLayer extends RNNLayer {
	private static final long serialVersionUID = 1L;
	BiRNNNeuroVo [] vos1;
	public BiRNNLayer(int nodeNN, 
			int t, boolean inHidenLayer, int previousNNN, int nextLayerNN) {
		super(nodeNN, t, inHidenLayer, previousNNN, nextLayerNN);
		vos1 = new BiRNNNeuroVo[vos0.length];
		for (int i = 0; i < vos1.length; i++) {
			vos1[i] = new BiRNNNeuroVo(vos0[i]);
		}
	}
	
	public void reverse(int lt) {
		for (int i = 0; i < vos1.length; i++) {
			vos1[i].reverse(lt);
		} 
	}
	
	public void reverseBack() {
		for (int i = 0; i < vos1.length; i++) {
			vos1[i].reverseBack();
		} 
	}
	
	public RNNNeuroVo[] getRNNNeuroVos() { 
		return vos1;
	}

}
