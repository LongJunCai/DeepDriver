package deepDriver.dl.aml.lstm.apps.pos.test;

import java.util.List;
import java.util.Properties;

import deepDriver.dl.aml.lstm.apps.pos.PosTagger;
import deepDriver.dl.aml.lstm.apps.util.TaggedWord;

/**
 * Demo for Pos Tagger predict method API
 * */

public class PosTaggerTest {
	
	public static void main(String[] args) {

		Properties prop = new Properties();
		prop.setProperty("dictFile", "C:/workspace/DeepDriver/models/POS/199801_dict.txt");
		prop.setProperty("sqFile", "C:/workspace/DeepDriver/models/POS/china_daily_1472638305564_3.m");
		
		String s1 = "新华社 北京 十二月 二十五日 电 （ 记者  ）";
		PosTagger tagger = new PosTagger(prop);
		List<TaggedWord> line = tagger.predict(s1);
		System.out.println(line.toString());
		
	}
}
