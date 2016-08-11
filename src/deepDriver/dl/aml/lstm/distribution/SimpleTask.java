package deepDriver.dl.aml.lstm.distribution;

import java.io.Serializable;

import deepDriver.dl.aml.distribution.ITask;

public class SimpleTask implements ITask, Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	int start;
	int end;
	int mbatch;
	
	public SimpleTask(int start, int end, int mbatch) {
		super();
		this.start = start;
		this.end = end;
		this.mbatch = mbatch;
	}
	public int getStart() {
		return start;
	}
	public void setStart(int start) {
		this.start = start;
	}
	public int getEnd() {
		return end;
	}
	public void setEnd(int end) {
		this.end = end;
	}
	public int getMbatch() {
		return mbatch;
	}
	public void setMbatch(int mbatch) {
		this.mbatch = mbatch;
	}
	
}
