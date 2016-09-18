package deepDriver.dl.aml.lstm.apps.util;

/**
 * Util Class for Labeling Tagged Word [word/tag]
 * */

public class TaggedWord {

	private String word;
	private String tag;

	private static final String DIVIDER = "/";

	public TaggedWord() {
		super();
	}

	public TaggedWord(String word) {
		this.word = word;
	}

	public TaggedWord(String word, String tag) {
		this.word = word;
		this.tag = tag;
	}

	public String tag() {
		return tag;
	}

	public void setTag(String tag) {
		this.tag = tag;
	}

	public String word() {
		return word;
	}

	public void setWord(String word) {
		this.word = word;
	}	
	
	@Override
	public String toString() {
		return toString(DIVIDER);
	}

	public String toString(String divider) {
		return word + divider + tag;
	}

	public void setFromString(String taggedWord) {
		setFromString(taggedWord, DIVIDER);
	}

	public void setFromString(String taggedWord, String divider) {
		int where = taggedWord.lastIndexOf(divider);
		if (where >= 0) {
			setWord(taggedWord.substring(0, where));
			setTag(taggedWord.substring(where + 1));
		} else {
			setWord(taggedWord);
			setTag(null);
		}
	}
	
}
