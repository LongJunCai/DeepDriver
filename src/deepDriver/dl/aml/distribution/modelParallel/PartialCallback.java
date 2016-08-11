package deepDriver.dl.aml.distribution.modelParallel;

public interface PartialCallback {
	public void runPartial(int offset, int runLen);
}