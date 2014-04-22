import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.Arrays;
import java.util.*;

public class AsteroidConverter
{
    static boolean debug = true;
    static boolean visualize = false;
    static String execCommand = null;
    static String trainFile = null;
    static String testFile = null;
    static String folder = "";

    public void printMessage(String s) {
        if (debug) {
            System.out.println(s);
        }
    }

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

    public void doExec() throws Exception {

        // read training file
        printMessage("Loading files..");
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
          //      printMessage(folder + s + ".det loaded. Rows = " + detTraining.size());

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

                // call trainingData(imageData, detections)
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
                String trainResp = reader.readLine();
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
                String testResp = reader.readLine();

            }
            br.close();
        }

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
            } else {
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

