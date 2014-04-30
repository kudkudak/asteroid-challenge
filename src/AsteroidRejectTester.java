import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.Arrays;
import java.util.*;

public class AsteroidRejectTester
{
    static boolean debug = true;
    static boolean visualize = false;
    static String execCommand = null;
    static String trainFile = null;
    static String testFile = null;
    static String folder = "";


    static boolean to_file = false;
    static String filename = "";

    public void printMessage(String s) {
        if (debug) {
            System.out.println(s);
        }
    }

    static final String _ = File.separator;


    // Score a testcase, given detections and rejections and user answers
    public double scoreAnswer(int[] userAns, Set<Integer> modelAnsDetect, Set<Integer> modelAnsReject) throws Exception
    {
        double score = 0.0;
        double total = 0;
        double correct = 0;
        Set<Integer> userAnsUsed = new TreeSet<Integer>();
        for (int i=0;i<userAns.length;i++)
        {
            total += 1.0;
            int id = userAns[i];
            if (!modelAnsDetect.contains(id) && !modelAnsReject.contains(id))
            {
                printMessage("Unique ID " +id + " not valid.");
                return 0.0;
            }
            if (userAnsUsed.contains(id))
            {
                printMessage("Unique ID " +id +" already used.");
                return 0.0;
            }
            userAnsUsed.add(id);
            if (modelAnsDetect.contains(id))
            {
                correct += 1.0;
                printMessage("1");
                score += (1000000.0 / modelAnsDetect.size()) * (correct / total);
            }
            printMessage("0");
        }

        return score;
    }
    

    public void visualize(ArrayList<Integer> raw, int offset, String fileName) throws Exception
    {
            int W = (64+10)*4-10;
            BufferedImage bi = new BufferedImage(W, 64, 1);
            Graphics2D g = (Graphics2D)bi.getGraphics();
            for (int y=0;y<64;y++)
                for (int x=0;x<W;x++)
                    bi.setRGB(x, y, 0xffffff);
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
                    int ival = raw.get(off++);
                    double dval = (double)(ival) / 256.0;
                    if (dval<dmin) ival = 0;
                    else if (dval>dmax) ival = 255;
                    else
                    {
                        dval = Math.max(0.0, Math.min(dval-dmin, dmax - dmin));
                        double d = ((Math.log(dval * linearF)) / log10) * logF;
                        ival = (int)(d);
                    }
                    if (ival<0) ival = 0;
                    if (ival>255) ival = 255;
                    int cr = ival;
                    int cg = ival;
                    int cb = ival;
                    int rgb = cr + (cg<<8) + (cb<<16);
                    bi.setRGB(x+(i*74),y,rgb);
                }
            }
            ImageIO.write(bi, "PNG", new File(fileName));
    }

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

    public void doExec() throws Exception {
        // launch solution
        printMessage("Executing your solution: " + execCommand + ".");
        Process solution = Runtime.getRuntime().exec(execCommand);

        BufferedReader reader = new BufferedReader(new InputStreamReader(solution.getInputStream()));
        PrintWriter writer = new PrintWriter(solution.getOutputStream());

        if(to_file){
            File file = new File(filename);
            writer = new PrintWriter(file);
            printMessage("Writing to file");
        } 

        new ErrorStreamRedirector(solution.getErrorStream()).start();

        // read training file
        printMessage("Training...");
        int det_id = 0;
        int num_train_rjct = 0;
        {
            BufferedReader br = new BufferedReader(new FileReader(trainFile));
            while (true) {
                String s = br.readLine();
                //printMessage(s);
                if (s == null) {
                    break;
                }
                // load raw image data
                ArrayList<Integer> rawTraining = new ArrayList<Integer>();
                loadRawImage(folder + s + ".raw", rawTraining);
                // load detection data
                ArrayList<String> detTraining = new ArrayList<String>();
                BufferedReader brdet = new BufferedReader(new FileReader(folder + s + ".det"));
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
                        num_train_rjct++;
                        trainAns.add(det_id);
                    }
                    detTraining.add(row);
                    cnt++;
                    if ((cnt%4)==0) det_id++;
                }
                brdet.close();
                printMessage(folder + s + ".det loaded. Rows = " + detTraining.size());

                if (visualize)
                {
                    int n = rawTraining.size()/(4*64*64);
                    for (int i=0;i<n;i++)
                    {
                        int case_num = det_id - n + i;
                        String fileName = case_num + ".png";
                        if (trainAns.contains(case_num))
                            fileName = "R_" + fileName;
                        else
                            fileName = "D_" + fileName;
                        visualize(rawTraining, i*4*64*64, fileName);
                    }
                }

                printMessage("call trainingData(imageData, detections)");
                int[] imageData_train = new int[rawTraining.size()];
                for (int i=0;i<rawTraining.size();i++)
                    imageData_train[i] = rawTraining.get(i);
                String[] detections_train = new String[detTraining.size()];
                detTraining.toArray(detections_train);

                writer.println(imageData_train.length);
                for (int v : imageData_train) {
                    writer.println(v);
                }
                writer.flush();

                writer.println(detections_train.length);
                for (String v : detections_train) {
                    writer.println(v);
                }
                writer.flush();

                // get response from solution
                if(!to_file){
                    String trainResp = reader.readLine();
                    printMessage("Got training response "+trainResp);
                }
            }
            br.close();
        }

        // read testing file
        printMessage("Testing...");
        Set<Integer> modelAnsReject = new TreeSet<Integer>();
        Set<Integer> modelAnsDetect = new TreeSet<Integer>();
        {
            BufferedReader br = new BufferedReader(new FileReader(testFile));
            while (true) {
                String s = br.readLine();
                //printMessage(s);
                if (s == null) {
                    break;
                }
                // load raw image data
                ArrayList<Integer> rawTest = new ArrayList<Integer>();
                loadRawImage(folder + s + ".raw", rawTest);
                // load detection data
                ArrayList<String> detTest = new ArrayList<String>();
                BufferedReader brdet = new BufferedReader(new FileReader(folder + s + ".det"));
                int cnt = 0;
                while (true) {
                    String row = brdet.readLine();
                    if (row == null) {
                        break;
                    }
                    row = det_id + " " + row;
                    if (row.charAt(row.length()-1)=='1')
                    {
                        modelAnsReject.add(det_id);
                    } else
                    {
                        modelAnsDetect.add(det_id);
                    }
                    // remove truth
                    row = row.substring(0, row.length()-2);
                    detTest.add(row);
                    cnt++;
                    if ((cnt%4)==0)
                        det_id++;
                }
                brdet.close();

                if (visualize)
                {
                    int n = rawTest.size()/(4*64*64);
                    for (int i=0;i<n;i++)
                    {
                        int case_num = det_id - n + i;
                        String fileName = case_num + ".png";
                        if (modelAnsReject.contains(case_num))
                            fileName = "R_" + fileName;
                        else
                            fileName = "D_" + fileName;
                        visualize(rawTest, i*4*64*64, fileName);
                    }
                }


                // call testData(imageData, detections)
                int[] imageData_test = new int[rawTest.size()];
                for (int i=0;i<rawTest.size();i++)
                    imageData_test[i] = rawTest.get(i);
                String[] detections_test = new String[detTest.size()];
                detTest.toArray(detections_test);

                writer.println(imageData_test.length);
                for (int v : imageData_test) {
                    writer.println(v);
                }
                writer.flush();

                writer.println(detections_test.length);
                for (String v : detections_test) {
                    writer.println(v);
                }
                writer.flush();

                // get response from solution
                if(!to_file){
                    String testResp = reader.readLine();
                }

            }
            br.close();
        }
        printMessage("Done testing");
        if(!to_file){
        // get response from solution
        String cmd = reader.readLine();
        int n = Integer.parseInt(cmd);
        if (n!=modelAnsReject.size()+modelAnsDetect.size())
        {
            printMessage("Invalid number of detections in return. " + (modelAnsReject.size()+modelAnsDetect.size()) + " expected, but " + n + " in list.");
            printMessage("Score = 0");
        }
        int[] userAns = new int[n];
        for (int i=0;i<n;i++) {
            String val = reader.readLine();
            userAns[i] = Integer.parseInt(val);
        }

        // call scoring function
        double score = scoreAnswer(userAns, modelAnsDetect, modelAnsReject);
        printMessage("Score = " + score);
        }
    }


    public static void main(String[] args) throws Exception {


       for (int i = 0; i < args.length; i++) {
            if (args[i].equals("-train")) {
                trainFile = args[++i];
            } else if (args[i].equals("-test")) {
                testFile = args[++i];
            } else if (args[i].equals("-exec")) {
                execCommand = args[++i];
            } else if (args[i].equals("-silent")) {
                debug = false;
            } else if (args[i].equals("-folder")) {
                folder = args[++i];
            } else if (args[i].equals("-vis")) {
                visualize = true;
            } else if (args[i].equals("-to_file")) {
                to_file = true;
                filename = args[++i];                
            }
            else {
                System.out.println("WARNING: unknown argument " + args[i] + ".");
            }
        }

        try {
            if (trainFile != null && testFile != null && execCommand != null) {
                new AsteroidRejectTester().doExec();
            } else {
                System.out.println("WARNING: nothing to do for this combination of arguments.");
            }
        } catch (Exception e) {
            System.out.println("FAILURE: " + e.getMessage());
            e.printStackTrace();
        }
    }

    class ErrorStreamRedirector extends Thread {
        public BufferedReader reader;

        public ErrorStreamRedirector(InputStream is) {
            reader = new BufferedReader(new InputStreamReader(is));
        }

        public void run() {
            while (true) {
                String s;
                try {
                    s = reader.readLine();
                } catch (Exception e) {
                    // e.printStackTrace();
                    printMessage("Exception in reading error stream "+e.toString());
                    return;
                }
                if (s == null) {
                    break;
                }
                System.out.println(s);
            }
        }
    }
}

