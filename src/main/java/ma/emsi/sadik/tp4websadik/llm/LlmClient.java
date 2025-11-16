package ma.emsi.sadik.tp4websadik.llm;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.enterprise.context.Dependent;

import java.io.Serializable;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Classe métier qui gère la connexion à Gemini via LangChain4j.
 * VERSION 1 : RAG de base (Test 2) avec 2 PDF.
 */
@Dependent
public class LlmClient implements Serializable {

    private String systemRole;
    private Assistant assistant;
    private ChatMemory chatMemory;

    /**
     * Interface définissant l’interaction "chat".
     */
    public interface Assistant {
        String chat(String prompt);
    }

    public LlmClient() {
        // --- Fonctionnalité Test 2 : Logging ---
        configureLogger();

        String apiKey = System.getenv("GEMINI_API_KEY");
        if (apiKey == null) {
            throw new RuntimeException("Variable d'environnement GEMINI_API_KEY non définie");
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true)
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        //Ingérer les 2 documents PDF
        try {
            List<String> documentNames = List.of("/rag.pdf", "/ml.pdf");
            ingestDocuments(documentNames, new ApacheTikaDocumentParser(), embeddingModel, embeddingStore);
        } catch (Exception e) {
            throw new RuntimeException("Erreur lors de l'ingestion des documents RAG", e);
        }

        System.out.println("Phase 1 (Ingestion RAG) terminée.");

        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever) // Ajout du RAG
                .build();
    }


    /**
     * Charge, parse, segmente et intègre une liste de documents dans le EmbeddingStore.
     */
    private void ingestDocuments(List<String> resourceNames,
                                 DocumentParser parser,
                                 EmbeddingModel embeddingModel,
                                 EmbeddingStore<TextSegment> embeddingStore) throws URISyntaxException {

        DocumentSplitter splitter = DocumentSplitters.recursive(600, 0);
        List<TextSegment> allSegments = new ArrayList<>();
        List<Embedding> allEmbeddings = new ArrayList<>();

        for (String resourceName : resourceNames) {
            System.out.println("Ingestion de : " + resourceName);
            URL fileUrl = LlmClient.class.getResource(resourceName);
            if (fileUrl == null) {
                System.err.println("Erreur: Fichier ressource non trouvé : " + resourceName);
                continue;
            }

            Path path = Paths.get(fileUrl.toURI());
            Document document = FileSystemDocumentLoader.loadDocument(path, parser);
            List<TextSegment> segments = splitter.split(document);

            Response<List<Embedding>> response = embeddingModel.embedAll(segments);

            allSegments.addAll(segments);
            allEmbeddings.addAll(response.content());
            System.out.println("Ingestion terminée pour : " + resourceName);
        }

        if (!allSegments.isEmpty()) {
            embeddingStore.addAll(allEmbeddings, allSegments);
        }
    }

    /**
     * Configure le logger pour dev.langchain4j (Fonctionnalité Test 2).
     */
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }


    /**
     * Setter appelé lorsque l'utilisateur choisit le rôle système.
     */
    public void setSystemRole(String role) {
        this.systemRole = role;
        chatMemory.clear();
        chatMemory.add(SystemMessage.from(role));
    }

    /**
     * Envoie la question au LLM et retourne la réponse.
     */
    public String ask(String question) {
        return assistant.chat(question);
    }
}