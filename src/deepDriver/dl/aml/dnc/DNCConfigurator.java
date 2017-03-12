package deepDriver.dl.aml.dnc;

import deepDriver.dl.aml.ann.ANN;
import deepDriver.dl.aml.lstm.LSTMConfigurator;

public class DNCConfigurator {
	
	DNCMemory memory;
	DNCReadHead [] readHeads;
	DNCWriteHead writeHead;
	DNCController controller;	
	
	int yLen = 30;//the merged y, before outputing.
	int rhNum;
	int memoryNum;
	int memoryLength;
	
	int maxTime = 100;
	
	int trainingLoop = 100000;
	
	int ldecayLoop = 20000;
	
	public DNCConfigurator(double l, double m, int maxTime, ANN ann, LSTMConfigurator cfg, int yLen, int rhNum, int memoryNum, int memoryLength) { 
		this.l = l;
		this.m = m;
		this.yLen = yLen;
		this.rhNum = rhNum;
		this.memoryNum = memoryNum;
		this.memoryLength = memoryLength;
		
		this.maxTime = maxTime;
		
		controller = new DNCController(ann, cfg, this); 
		
		memory = new DNCMemory(memoryNum, memoryLength, this);
		
		readHeads = new DNCReadHead[rhNum];
		for (int i = 0; i < readHeads.length; i++) {
			readHeads[i] = new DNCReadHead(this);
		}
		
		writeHead = new DNCWriteHead(this);
	}

	public int getMaxTime() {
		return maxTime;
	}

	public void setMaxTime(int maxTime) {
		this.maxTime = maxTime;
	}
	
	double l = 0.001;
	double m = 0.1;
	
	double ml = 0.0001;	

	public double getMl() {
		return ml;
	}

	public void setMl(double ml) {
		this.ml = ml;
	}

	public double getL() {
		return l;
	}

	public void setL(double l) {
		this.l = l;
	}

	public double getM() {
		return m;
	}

	public void setM(double m) {
		this.m = m;
	}

	public int getLdecayLoop() {
		return ldecayLoop;
	}

	public void setLdecayLoop(int ldecayLoop) {
		this.ldecayLoop = ldecayLoop;
	}
	
}
