import java.io.IOException;
import java.io.FileReader;
import java.io.FileNotFoundException;
import java.nio.file.Paths;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field.Store;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.util.List;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.Set;
import java.util.HashSet;
import java.lang.Math;

public class IndexCreate {
    public static void main(String[] args) {
        List<String> data_modes = new ArrayList<>();
        data_modes.add("train");
        data_modes.add("validation");
        data_modes.add("test");
        
        for (int k = 0; k < data_modes.size(); k++){
            System.out.println("create dailydialog candidates index of " + data_modes.get(k));

            String storage_path = "./index/dailydialog_candidates_" + data_modes.get(k);
            String candidate_response_path = "../data/DailyDialog_tmp/" + data_modes.get(k) + "/pair-1/original_dialog_response_uni.text";

            Analyzer analyzer = new StandardAnalyzer();
            IndexWriterConfig indexWriterConfig = new IndexWriterConfig(analyzer);
            indexWriterConfig.setOpenMode(OpenMode.CREATE_OR_APPEND);

            Directory directory = null;
            IndexWriter indexWriter = null;
            try {
                directory = FSDirectory.open(Paths.get(storage_path));
                indexWriter = new IndexWriter(directory, indexWriterConfig);
            } catch (IOException e) {
                e.printStackTrace();
            }

            System.out.println("Loading dailydialog candidates.....");
            Scanner sc = null;
            List<String> allLines = new ArrayList<>();
            String file = candidate_response_path;
            Set<String> hast_set = new HashSet<>();
            try {
                sc = new Scanner(new FileReader(file));

                int i = 0;
                while (sc.hasNextLine()) {
                    i++;
                    String text = sc.nextLine();
                    if (hast_set.contains(text)){
                        continue;
                    }else{
                        hast_set.add(text);
                        allLines.add(text);
                    }
                }
            } catch (FileNotFoundException e) {
                System.out.println("Input file " + file + " not found");
                System.exit(1);
            } finally {
                sc.close();
            }

            System.out.println("\ncreate dailydialog candidates docs.....");
            List<Document>docs = new ArrayList<>();
            for (int i = 0; i < allLines.size(); i++){
                Document doc = new Document();
                doc.add(new TextField("content", allLines.get(i), Store.YES));
                docs.add(i, doc);
            }

            try {
                System.out.println("Add dailydialog candidates docs into Index.....");
                for (int i = 0; i < docs.size(); i++){
                    indexWriter.addDocument(docs.get(i));
                }

                indexWriter.commit();
            } catch (IOException e) {
                e.printStackTrace();
            } finally {
                try {
                    indexWriter.close();
                    directory.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}