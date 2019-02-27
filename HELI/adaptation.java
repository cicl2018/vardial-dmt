package main;

import com.google.common.collect.HashBasedTable;

import java.util.*;
import java.io.*;

public class adaptation {
    public static String address = "/home/nianheng/Documents/dmt/n=8_p=3.87_+iteration/";
    public Map<Integer, List> sentenceConfidence = new HashMap();
    public IndexMinPQ<Double> pq;
    public int initial = 2000;

    public void addSentence(String sentence, String winnerLang, Double guoyuScore, Double mandarinScore, int count){
        Double confidence = 0.0;
        if (winnerLang.equals("guoyu")){
            confidence = guoyuScore - mandarinScore;
        }
        else if (winnerLang.equals("mandarin")){
            confidence = mandarinScore - guoyuScore;
        }
        ArrayList<String> infoList = new ArrayList<>();
        infoList.add(sentence);
        infoList.add(confidence.toString());
        infoList.add(guoyuScore.toString());
        infoList.add(mandarinScore.toString());
        infoList.add(winnerLang);
        sentenceConfidence.put(count,infoList);

    }
    public void createPQ(){
        pq = new IndexMinPQ<>(initial);
        for (Map.Entry<Integer, List> entry : sentenceConfidence.entrySet()){
            pq.insert(entry.getKey(), Double.valueOf((String)entry.getValue().get(1)));
        }
    }

    public int bestSentence(){
       int bestSentenceID = -1;
        if (!pq.isEmpty()) {
            bestSentenceID = pq.delMin();
        }
        return bestSentenceID;
    }

    public void anAdaptation(){
        createPQ();
        createmodels adaptedModel = new createmodels();
        HeLI aHeli = new HeLI();
        int index = 0;
        String identifiedLanguage = null;
        String testLine = null;
        while (!(index== -1)){
            index = bestSentence();
            System.out.println(pq.size());
            System.out.println(sentenceConfidence.size());
            String bestSentence = (String) sentenceConfidence.get(index).get(0);
            String winnerLang = (String) sentenceConfidence.get(index).get(4);

            try(FileWriter fw = new FileWriter(address + "Test/result.txt", false);
                BufferedWriter bw = new BufferedWriter(fw);
                PrintWriter out = new PrintWriter(bw)){
                out.println("");
            } catch (IOException e) {}

            for(Iterator<Map.Entry<Integer, List>> it = sentenceConfidence.entrySet().iterator(); it.hasNext(); ) {
                Map.Entry<Integer, List> entry = it.next();
                if(entry.getKey().equals(index)) {
                    it.remove();
                }
            }
            if (winnerLang.equals("guoyu")){
                try(FileWriter fw = new FileWriter(address + "Training/guoyu.train", true);
                    BufferedWriter bw = new BufferedWriter(fw);
                    PrintWriter out = new PrintWriter(bw)){
                    out.println(bestSentence);
                } catch (IOException e) {}
            }
            else if (winnerLang.equals("mandarin")){
                try(FileWriter fw = new FileWriter(address + "Training/mandarin.train", true);
                    BufferedWriter bw = new BufferedWriter(fw);
                    PrintWriter out = new PrintWriter(bw)){
                    out.println(bestSentence);
                } catch (IOException e) {}
            }

            File file = new File(address + "Training/");

            File[] files = file.listFiles();

            for (File file2 : files) {
                if (file2.getName().contains(".train")) {
                    adaptedModel.createmodel(file2);
                }
            }

            File lan_file = new File(address + "languagelist");
            BufferedReader reader = null;

            try {
                reader = new BufferedReader(new FileReader(lan_file));
                String text = null;
                while ((text = reader.readLine()) != null) {
                    aHeli.languageList.add(text);
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    if (reader != null) {
                        reader.close();
                    }
                } catch (IOException e) {
                }
            }

            aHeli.gramDictLow = HashBasedTable.create();
            aHeli.gramDictCap = HashBasedTable.create();
            aHeli.wordDictCap = HashBasedTable.create();
            aHeli.wordDictLow = HashBasedTable.create();

            ListIterator gramiterator = aHeli.languageList.listIterator();
            while(gramiterator.hasNext()) {
                Object element = gramiterator.next();
                String language = (String) element;

                aHeli.loadIn(aHeli.usedlow1gram, language, "LowGramModel1");
                aHeli.loadIn(aHeli.usedlow2gram, language, "LowGramModel2");
                aHeli.loadIn(aHeli.usedlow3gram, language, "LowGramModel3");
                aHeli.loadIn(aHeli.usedlow4gram, language, "LowGramModel4");
                aHeli.loadIn(aHeli.usedlow5gram, language, "LowGramModel5");
                aHeli.loadIn(aHeli.usedlow6gram, language, "LowGramModel6");
                aHeli.loadIn(aHeli.usedlow7gram, language, "LowGramModel7");
                aHeli.loadIn(aHeli.usedlow8gram, language, "LowGramModel8");

                aHeli.loadIn(aHeli.usedcap1gram, language, "CapGramModel1");
                aHeli.loadIn(aHeli.usedcap2gram, language, "CapGramModel2");
                aHeli.loadIn(aHeli.usedcap3gram, language, "CapGramModel3");
                aHeli.loadIn(aHeli.usedcap4gram, language, "CapGramModel4");
                aHeli.loadIn(aHeli.usedcap5gram, language, "CapGramModel5");
                aHeli.loadIn(aHeli.usedcap6gram, language, "CapGramModel6");
                aHeli.loadIn(aHeli.usedcap7gram, language, "CapGramModel7");
                aHeli.loadIn(aHeli.usedcap8gram, language, "CapGramModel8");

                aHeli.loadIn(aHeli.usedcapwords, language, "CapWordModel");
                aHeli.loadIn(aHeli.usedlowwords, language, "LowWordModel");
            }
            for (Map.Entry<Integer, List> entry : sentenceConfidence.entrySet()){
                testLine = (String)entry.getValue().get(0);
                int testIndex = entry.getKey();
                if (testLine.length() < 2) {
                    break;
                }
                String mysterytext = testLine;
                int count = 0;
                String[] result = aHeli.identifyText(mysterytext, count, false);
                Double guoyuScore = Double.valueOf(result[1]);
                Double mandarinScore = Double.valueOf(result[2]);
                identifiedLanguage = result[0];
                Double confidence = 0.0;
                if (identifiedLanguage.equals("guoyu")){
                    confidence = guoyuScore - mandarinScore;
                }
                else if (identifiedLanguage.equals("mandarin")){
                    confidence = mandarinScore - guoyuScore;
                }
                sentenceConfidence.get(testIndex).set(1, confidence.toString());
                sentenceConfidence.get(testIndex).set(2, guoyuScore.toString());
                sentenceConfidence.get(testIndex).set(3, mandarinScore.toString());
                count ++;
                try(FileWriter fw = new FileWriter(address + "Test/result.txt", true);
                    BufferedWriter bw = new BufferedWriter(fw);
                    PrintWriter out = new PrintWriter(bw)){
                    out.println(testLine+"\t"+identifiedLanguage);
                } catch (IOException e) {}
            }
            createPQ();
        }

    }
}
