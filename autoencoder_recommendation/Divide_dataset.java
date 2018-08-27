import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class Divide_dataset {
        public static void main(String[] args) throws IOException {
        String fileName = "/Users/chenxijia/personal/code/laiData/TFSample/autoencoder-master/ml-1m/ratings.dat";


        readUsingBufferedReader(fileName);
    }
    private static void readUsingBufferedReader(String fileName) throws     IOException {
        File file = new File(fileName);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String line;
        double i = 0;
        File train = new File("/Users/chenxijia/personal/train.dat");
        File test = new File("/Users/chenxijia/personal/test.dat");
        train.createNewFile();
        test.createNewFile();
        BufferedWriter out1 = new BufferedWriter(new FileWriter(train));
        BufferedWriter out2 = new BufferedWriter(new FileWriter(test));
        while((line = br.readLine()) != null){
        		i = Math.random()*5;
        		if(i>4) {
        			 out2.write(line+"\r\n");
        		}else out1.write(line+"\r\n"); 
        }
        //close resources
        br.close();
        fr.close();
    }
}
