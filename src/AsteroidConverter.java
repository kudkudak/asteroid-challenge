import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class AsteroidConverter
{
    static boolean debug = true;
    static boolean visualize = false;
    static String execCommand = null;
    static String trainFile = null;
    static String testFile = null;
    static String folder = "";

    public static void printMessage(String s) {
        if (debug) {
            System.out.println(s);
        }
    }

    static final int ImageSide = 64, ImageChannels = 4;

    static final String _ = File.separator;

    public void loadRawImage(String filename, ArrayList<Integer> raw) throws Exception {
        ObjectInputStream fi = new ObjectInputStream(new FileInputStream(filename));
        byte[] rawbytes = (byte[]) fi.readObject();
        for (int i=0;i+1<rawbytes.length;i+=2)
        {
            int v = (int)(rawbytes[i]&0xFF);
            v |= (int)(rawbytes[i+1]&0xFF) << 8;
            raw.add(v);
        }
     //   printMessage(filename + " loaded. Size = " + raw.size());
    }

    public void writeDataset() throws Exception {

        // read training file
        printMessage("Loading files..");
        File dir = new File("data");
        File[] directoryListing = dir.listFiles();
        int det_id = 0;
        int file_counter = 0;
        for (File child : directoryListing)
        {
                String filename = child.getAbsolutePath();
                String basename = filename.substring(0, filename.lastIndexOf('.')),
                        extension = filename.substring(filename.lastIndexOf('.') + 1);
                printMessage("Reading "+filename +" basename "+extension) ;

                // Skip not interesting files
                if(!extension.equals("raw")) continue;
                // load raw image data
                ArrayList<Integer> rawTraining = new ArrayList<Integer>();
                loadRawImage(child.getAbsolutePath(), rawTraining);
                // load detection data
                List<String> detTraining = new ArrayList<String>();
                BufferedReader brdet =
                        new BufferedReader(new FileReader(basename + ".det"));

                int cnt = 0;
                Set<Integer> trainAns = new TreeSet<Integer>();
                while (true) {
                    String row = brdet.readLine();
                    if (row == null) {
                        break;
                    }
                    row = det_id + " " + row;
                    if (row.charAt(row.length()-1)=='1')
                    {
                        trainAns.add(det_id);
                    }
                    detTraining.add(row);
                    cnt++;
                    if ((cnt%4)==0) det_id++;
                }
                brdet.close();


                int n = rawTraining.size()/(ImageChannels*ImageSide*ImageSide);
                for(int i=0;i<n;++i){
                    file_counter += 1;
                    File file = new File("trainer/data/"+file_counter+"_img.raw");
                    PrintWriter writer = new PrintWriter(file);
                    int shift = i*ImageChannels*ImageSide*ImageSide;
                    StringBuilder sb = new StringBuilder();
                    for(int j=0;j<ImageChannels*ImageSide*ImageSide;++j)
                       sb.append(rawTraining.get(j+shift)+" ");

                    writer.write(sb.toString());
                    writer.close();

                    file = new File("trainer/data/"+file_counter+".det");
                    writer = new PrintWriter(file);
                    writer.write(detTraining.get(i));
                    //writer.write(detTraining.stream().collect(Collectors.joining(" ")));
                    writer.close();
                }



                printMessage("Contains information about "+n+" detections");


        }
    }


    public static void main(String[] args){
        try{
            new AsteroidConverter().writeDataset();
        }
        catch(Exception e){
            printMessage("Exception "+e.toString());
        }
    }


}

