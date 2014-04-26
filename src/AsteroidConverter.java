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
    static boolean convert_log=true;

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

    public void convertLog(ArrayList<Integer> raw, int offset, ArrayList<Double> out) throws Exception
    {
           int W = (64+10)*4-10;
           for (int i=0;i<4;i++)
            {
                int off = offset + i*64*64;
                int imin = 1<<20;
                int imax = -imin;
                // Find min and max
                for (int j=0;j<4096;j++)
                {
                    int r = raw.get(j+off);
                    if (r>65500) continue;
                    imin = Math.min(imin, r);
                    imax = Math.max(imax, r);
                }
                double dmax = (double)(imax) / 256.0;
                double dmin = (double)(imin) / 256.0;
                if (dmax*0.5-dmin > 10)
                {
                    dmax *= 0.5;
                }
                if (dmax-dmin<0.0001) dmax += 0.1;

                double linearF = 255.0 / (dmax - dmin);
                double log10 = Math.log(10.0);
                double logF = 255.0 / (Math.log(255.0) / log10);
                for (int y=0;y<64;y++)
                for (int x=0;x<64;x++)
                {
                    double ival = (double)raw.get(off++);
                    double dval = (double)(ival) / 256.0;
                    if (dval<dmin) ival = 0;
                    else if (dval>dmax) ival = 255;
                    else
                    {
                        dval = Math.max(0.0, Math.min(dval-dmin, dmax - dmin));
                        double d = ((Math.log(dval * linearF)) / log10) * logF;
                        ival = d;
                    }
                    if (ival<0) ival = 0;
                    if (ival>255) ival = 255;
                    out.add(ival);
                }
            }
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
                printMessage("Considering next file");

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
                    if(convert_log==true){
                        ArrayList<Double> out = new ArrayList<Double>();
                        convertLog(rawTraining, shift, out);
                        for(int j=0;j<ImageChannels*ImageSide*ImageSide;++j)
                            sb.append(out.get(j)+" ");
                    }
                    else{
                        for(int j=0;j<ImageChannels*ImageSide*ImageSide;++j)
                            sb.append(rawTraining.get(j+shift)+" ");
                    }

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

