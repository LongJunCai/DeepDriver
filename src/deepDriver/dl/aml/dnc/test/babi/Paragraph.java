package deepDriver.dl.aml.dnc.test.babi;

import java.util.ArrayList;
import java.util.List;

import deepDriver.dl.aml.string.Dictionary;

public class Paragraph {
	
	List<Integer> words = new ArrayList<Integer>();
	int cnt = -1;
	
	List<KP> answer = new ArrayList<KP>();
	Paragraph next;
	
	Dictionary dic;
	
	class KP {
		int k;
		int v;
		public KP(int k, int v) {
			super();
			this.k = k;
			this.v = v;
		}
		
	}
	
	public void addWord(int wd) {
		words.add(wd);
	}
	
	public void addAnswer(int wd) {
		answer.add(new KP(wd, words.size() - 1));
	}
	
	boolean out = false;
	
	public int getAnswer() {
		if (out) {
			String as = dic.getIntMap().get(answer.get(cnt).k);
			System.out.println("ANSWER:="+as);
		}		
		return answer.get(cnt).k;
	}
	
	public int [] nextTxt() {
		cnt ++; 
		if (cnt <= answer.size() - 1) {
			int wp = answer.get(cnt).v;
			return getTxt(wp);
		}
		return null;
	}
	
	public int [] getTxt(int wp) {
		StringBuffer sb = new StringBuffer();
		int [] txt = new int[wp + 1];
		for (int i = 0; i < txt.length; i++) {
			txt[i] = words.get(i);
			if (out) {
				sb.append(dic.getIntMap().get(txt[i]));
				sb.append(" ");
			}
		}
		if (out) {
			System.out.println(sb.toString());
		}		
		return txt;
	}
	
	public void reset() {
		cnt = -1;
	}

	public Paragraph getNext() {
		return next;
	}

	public void setNext(Paragraph next) {
		this.next = next;
	}
	
	public int [] getFullTxt() {
		int [] txt = new int[words.size()];
		for (int i = 0; i < txt.length; i++) {
			txt[i] = words.get(i);
		}
		return txt;
	}
	
	public int [] getFullAnswer() {
		int [] anInt = new int[answer.size()];
		for (int i = 0; i < anInt.length; i++) {
			anInt[i] = answer.get(i).k;
		}
		return anInt;
	}
	
	public int [] getFullAnswerPos() {
		int [] pos = new int[answer.size()];
		for (int i = 0; i < pos.length; i++) {
			pos[i] = answer.get(i).v;
		}
		return pos;
	}

}
