package deepDriver.dl.aml.lstm;

public class BiCell extends BiRNNNeuroVo implements ICell {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	ICell cell;
	public BiCell(RNNNeuroVo vo) {
		super(vo);
		cell = (ICell) vo;
	}

	@Override
	public double[] getSc() {
 		return cell.getSc();
	}

	@Override
	public void setSc(double[] sc) {
		cell.setSc(sc);
	}

	@Override
	public double[] getDeltaSc() {
		return cell.getDeltaSc();
	}

	@Override
	public void setDeltaSc(double[] deltaSc) {
		cell.setDeltaSc(deltaSc);
	}

	@Override
	public double[] getCZz() {
		return cell.getCZz();
	}

	@Override
	public void setCZz(double[] scZz) {
		cell.setCZz(scZz);
	}

	@Override
	public double[] getDeltaC() { 
		return cell.getDeltaC();
	}

	@Override
	public void setDeltaC(double[] deltaC) {
		cell.setDeltaC(deltaC);
	}

}
