package deepDriver.dl.aml.cnn.txt;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;


public class DataPreparation 
{
	/**
	 * 获取标注的情感极性
	 * @param lineText
	 * @return int polarity
	 */
	public String getTextLabel(String lineText )
	{
		String [] input = lineText.split("\t");
		String polarity = input[3];
		return polarity;
	}
	
	/**
	 * 将文件中的一行文本（一条微博或豆瓣评论）转化为向量(词向量的拼接)
	 * @param lineText
	 * @param word2vec
	 * @return StringBuffer
	 */
	public StringBuffer line2vec(String lineText, Word2vecUtil word2vec )
	{	
		String [] input = lineText.split("\t");
		String [] words = input[2].split(" ");
		DecimalFormat decimalFormat=new DecimalFormat(".00");
		StringBuffer result = new StringBuffer();
		for(String word: words)
		{
			float [] vector = word2vec.getWordVector(word);
			if(vector==null)
			{
				continue;
			}
			for(float dim : vector)
			{
			    result.append(decimalFormat.format(dim * 100));
			    result.append(',');
			}
		}
		return result;
	}
	
	
	  /**
     * 将文件中的一行文本（一条微博或豆瓣评论）转化为padding后向量(词向量的拼接)
     * @param lineText
     * @param word2vec
     * @return StringBuffer
     */
    public StringBuffer line2paddingVec(String lineText, int maxWordNum, Word2vecUtil word2vec )
    {   
        String [] input = lineText.split("\t");
        String [] words = input[1].split(" ");
        int sentenceLength = words.length;
        int paddingLength = 0;
        DecimalFormat decimalFormat=new DecimalFormat(".00");
        StringBuffer result = new StringBuffer();
        
        if(sentenceLength <= maxWordNum)
        {
            paddingLength = maxWordNum - sentenceLength;
            for(String word: words)
            {
                float [] vector = word2vec.getWordVector(word);
                if(vector==null)
                {
                    paddingLength++;
                    continue;
                }
                for(float dim : vector)
                {
                    result.append(decimalFormat.format(dim * 100));
                    result.append(',');
                }
            }
            
            for(int padding = 0; padding < paddingLength; padding++)
            {
                for(int i = 0; i < 200 ; i++)
                {
                    result.append('0');
                    result.append(',');
                }
            }
            
        }
        
        else
        {
            int wordIndex = 0;
            for(String word: words)
            {
                if(wordIndex == maxWordNum)
                {
                    break;
                }
                float [] vector = word2vec.getWordVector(word);
                if(vector==null)
                {
                    continue;
                }
                for(float dim : vector)
                {
                    result.append(decimalFormat.format(dim * 100));
                    result.append(',');
                }
                wordIndex ++;
            }
        }
        return result;
    }
	
	
	
	
	
	
	/**
	 * 准备cnn格式的输入
	 * @param inputPath
	 * @param outputPath
	 * @param word2vec
	 * @return
	 * @throws IOException
	 */
	public void getTrainingSet(String inputPath, Word2vecUtil word2vec,String outputPath , int maxWordNum) throws IOException
	{	
	     DataPreparation dataPreparation = new DataPreparation();
	     
	     FileReader reader = new FileReader(inputPath);
         BufferedReader br = new BufferedReader(reader);
         
         PrintStream ps = new PrintStream(outputPath); 
         
         String str = null;
         while((str = br.readLine()) != null) 
         {
        	 StringBuffer newline = new StringBuffer();
        	 StringBuffer linevec = new StringBuffer();
        	 String label = dataPreparation.getTextLabel(str);
        	 newline.append(label+',');
        	 if(maxWordNum == -1)
        	 {
        	      linevec = dataPreparation.line2vec(str, word2vec);
        	 }
        	 else
        	 {
        	      linevec = dataPreparation.line2paddingVec(str, maxWordNum, word2vec);
        	 }
        	 if( linevec.length() > 0)
        	 {
        	     newline.append(linevec);
        	 }  
        	 else
        	 {
        	     continue;
        	 }
//        	 System.out.println(str);
        	 String res = newline.toString().substring(0, newline.lastIndexOf(","));
        	 System.out.println(res);
//        	 String [] input = newline.toString().split(",");
//        	 System.out.println((input.length-1)/200);
//        	 if((input.length-1)/200==20)
//        	 {
        	     ps.println( res); 
//        	 }
        	 
         }
         ps.close();
         br.close();
         reader.close();
	}
	
	public static void main(String[] args) throws IOException 
	{
		Word2vecUtil word2vec = new Word2vecUtil();
		word2vec.loadModel("E://data//word2vecResult//vectors.bin");
		String filepath = "E://models//cnn//sentiment3.0//";
		DataPreparation dataPreparation = new DataPreparation();
		
		dataPreparation.getTrainingSet(filepath + "test-douban-corpus.txt", word2vec, filepath + "input-test-douban-10000.txt",-1);
		System.out.println("ok");
		
		
		
//		 FileReader reader = new FileReader(filepath + "cnn-sentiment-test-6000-padding-1.txt");
//         BufferedReader br = new BufferedReader(reader);
//         String str = null;
//         while((str = br.readLine()) != null) 
//         {
//             String [] input = str.split(",");
//             System.out.println(input.length);
//         }
		
		
		
	}
	
}
