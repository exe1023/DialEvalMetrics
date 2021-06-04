import java.io.IOException;
import java.nio.file.Paths;
import java.io.PrintWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.search.SortField;

import java.util.List;
import java.util.ArrayList;
import java.util.Scanner;

import java.lang.Math;

public class IndexSearch{

    public static void main(String[] args) {
        List<String> data_modes = new ArrayList<>();
        data_modes.add("train");
        data_modes.add("validation");
        data_modes.add("test");
        
        for (int i = 0; i < data_modes.size(); i++){
            Directory directory = null;
            try {
                directory = FSDirectory.open(Paths.get("./index/dailydialog_candidates_"+data_modes.get(i))); 
                DirectoryReader directoryReader = DirectoryReader.open(directory);
                IndexSearcher searcher = new IndexSearcher(directoryReader);
                Analyzer analyzer = new StandardAnalyzer();

                String file = "../data/DailyDialog_tmp/";

                Scanner sc = null;
                List<String> allQueries = new ArrayList<>();
                PrintWriter output = null;
                String cur_file = "";

                sc = null;
                allQueries = new ArrayList<>();
                cur_file = file + data_modes.get(i) + "/pair-1/original_dialog_response.text";

                String o_filepath = file + data_modes.get(i) + "/original_dialog_lexical_neg.response";
                File o_file = new File(o_filepath);
                o_file.createNewFile();
                output = new PrintWriter(o_file);

                try {
                    sc = new Scanner(new FileReader(cur_file));

                    while (sc.hasNextLine()) {
                        String text = sc.nextLine();
                        allQueries.add(text);
                    }
                } catch (FileNotFoundException e) {
                    System.out.println("Input file " + cur_file + " not found");
                    System.exit(1);
                } finally {
                    sc.close();
                }


                QueryParser parser = new QueryParser("content", analyzer);
                String cur_five_negs = "";
                for (int q_idx=0; q_idx < allQueries.size(); q_idx++){
                    cur_five_negs = ""; 

                    Query query = parser.parse(parser.escape(allQueries.get(q_idx))); 
                    TopDocs topDocs = searcher.search(query, 1); 
                    int start = 0;
                    int end = 0;

                    if (topDocs.totalHits.value > 500000){
                        topDocs = searcher.search(query, 300000); 
                        start = 299995;
                        end = 299999;
                    }
                    else if (topDocs.totalHits.value > 100000){
                        topDocs = searcher.search(query, 70000); 
                        start = 69995;
                        end = 69999;
                    }
                    else if (topDocs.totalHits.value > 50000){
                        topDocs = searcher.search(query, 30000); 
                        start = 29995;
                        end = 29999;
                    }
                    else if (topDocs.totalHits.value > 10000){
                        topDocs = searcher.search(query, 7000); 
                        start = 6995;
                        end = 6999;
                    }
                    else if (topDocs.totalHits.value > 1000){
                        topDocs = searcher.search(query, 700); 
                        start = 695;
                        end = 699;
                    }
                    else if (topDocs.totalHits.value > 500){
                        topDocs = searcher.search(query, 200); 
                        start = 195;
                        end = 199;
                    }
                    else if (topDocs.totalHits.value > 200){
                        topDocs = searcher.search(query, 100); 
                        start = 95;
                        end = 99;
                    }
                    else if (topDocs.totalHits.value > 100){
                        topDocs = searcher.search(query, 50); 
                        start = 45;
                        end = 49;
                    }
                    else if (topDocs.totalHits.value > 50){
                        topDocs = searcher.search(query, 20); 
                        start = 15;
                        end = 19;
                    }
                    else if (topDocs.totalHits.value > 20){
                        topDocs = searcher.search(query, 15); 
                        start = 10;
                        end = 14;
                    }
                    else if (topDocs.totalHits.value > 5){
                        topDocs = searcher.search(query, 5); 
                        start = 0;
                        end = 4;
                    }
                    else{
                        topDocs = searcher.search(query, 1);
                        start = 0;
                        end = 0;
                    }

                    System.out.println(allQueries.get(q_idx));
                    if (topDocs.scoreDocs.length != 0) { 
                        for (int j = start; j <= end ; j++) { 
                            Document doc = searcher.doc(topDocs.scoreDocs[j].doc);
                            String cur_answer = doc.get("content"); 
                            System.out.println("content = " + cur_answer);
                            if (j!=end){ 
                                cur_five_negs += (cur_answer + "|||");
                            }
                            else{
                                cur_five_negs += cur_answer;
                            }
                        }
                        output.println(cur_five_negs);
                        System.out.println("\n");
                    }else{
                        output.println(cur_five_negs);
                        System.out.println("\n");
                    }
                }

                output.close(); 
                directory.close();
                directoryReader.close();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (ParseException e) {
                e.printStackTrace();
            }
        }
    }
}