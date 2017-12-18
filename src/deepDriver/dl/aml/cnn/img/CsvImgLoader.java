package deepDriver.dl.aml.cnn.img;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class CsvImgLoader {
	
	private List<String> imgs = new ArrayList<String>();
	String header;
	
	public boolean isHeader() {
        return isHeader;
    }
    public void setHeader(boolean isHeader) {
        this.isHeader = isHeader;
    }
    
    public int size() {
		return imgs.size();
	}
    
    public String get(int id) {
		return imgs.get(id);
	}
    
    public List<String> getImgs() {
		return imgs;
	}
	public void setImgs(List<String> imgs) {
		this.imgs = imgs;
	}


	boolean isHeader = true;
	public void loadImg(String file) throws Exception {
		BufferedReader bi = new BufferedReader( new InputStreamReader(new 
				FileInputStream(new File(file)), "utf-8"));
		String content = bi.readLine();
		while (content != null) {
			content = content.trim();
			if (content.length() == 0) {
				content = bi.readLine();
				continue;
			}
			if (isHeader) {
				isHeader = false;
				header = content;
				content = bi.readLine();
				continue;
			}
			imgs.add(content);
			content = bi.readLine();
		}
		bi.close();
	}

	public void loadSingle(String content) throws Exception 
	{
	    if(content != null)
	    {
	        content = content.trim();
	        imgs.add(content);
	    }
	}
}
