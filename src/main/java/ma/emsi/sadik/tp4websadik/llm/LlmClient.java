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
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import jakarta.enterprise.context.Dependent;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter; // <-- Test 3
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.io.Serializable;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap; // <-- Test 3
import java.util.List;
import java.util.Map; // <-- Test 3
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Classe métier qui gère la connexion à Gemini via LangChain4j.
 * VERSION 3 : RAG sémantique (Test 2 + Test 3 + Test 5).
 */
@Dependent
public class LlmClient implements Serializable {

    private String systemRole;
    private Assistant assistant;
    private ChatMemory chatMemory;

    public interface Assistant {
        String chat(String prompt);
    }

    public LlmClient() {
        // --- Fonctionnalité Test 2 : Logging ---
        configureLogger();

        // --- Configuration des Clés API ---
        String apiKey = System.getenv("GEMINI_API_KEY");
        String tavilyKey = System.getenv("TAVILY_API_KEY");
        if (apiKey == null || tavilyKey == null) {
            throw new RuntimeException("Clés API (GEMINI_API_KEY ou TAVILY_API_KEY) non définies");
        }

        // --- Modèle de Chat (LLM) ---
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true) // Test 2
                .temperature(0.3) // Test 3
                .build();

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // --- Source 1 : PDF "rag.pdf" (Test 3) ---
        EmbeddingStore<TextSegment> ragStore = ingestDocument("/rag.pdf", new ApacheTikaDocumentParser(), embeddingModel);
        ContentRetriever ragRetriever = EmbeddingStoreContentRetriever.from(ragStore);
        System.out.println("Ingestion de rag.pdf terminée.");

        // --- Source 2 : PDF "ml.pdf" (Test 3) ---
        EmbeddingStore<TextSegment> mlStore = ingestDocument("/ml.pdf", new ApacheTikaDocumentParser(), embeddingModel);
        ContentRetriever mlRetriever = EmbeddingStoreContentRetriever.from(mlStore);
        System.out.println("Ingestion de ml.pdf terminée.");

        // --- Source 3 : Web Search (Test 5) ---
        WebSearchEngine tavily = TavilyWebSearchEngine.builder().apiKey(tavilyKey).build();
        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(tavily)
                .maxResults(5)
                .build();

        Map<ContentRetriever, String> retrieverMap = new HashMap<>();

        retrieverMap.put(ragRetriever, "Informations sur RAG, LangChain4j, et l'IA (sujet du fichier rag.pdf)");
        retrieverMap.put(mlRetriever, "Informations sur le Machine Learning et les algorithmes (sujet du fichier ml.pdf)");
        retrieverMap.put(webRetriever, "Actualités, météo, sports, et informations générales en temps réel sur le Web");

        QueryRouter queryRouter = new LanguageModelQueryRouter(model, retrieverMap);

        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        this.chatMemory = MessageWindowChatMemory.withMaxMessages(10);

        this.assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .retrievalAugmentor(augmentor)
                .build();
    }

    /**
     * Nouvelle méthode d'ingestion (basée sur le Test 3).
     * Charge, parse, segmente UN SEUL document.
     */
    private EmbeddingStore<TextSegment> ingestDocument(String resourceName,
                                                       DocumentParser parser,
                                                       EmbeddingModel embeddingModel) {
        try {
            URL fileUrl = LlmClient.class.getResource(resourceName);
            if (fileUrl == null) {
                System.err.println("Erreur: Fichier ressource non trouvé : " + resourceName);
                throw new RuntimeException("Ressource non trouvée : " + resourceName);
            }
            Path path = Paths.get(fileUrl.toURI());
            Document document = FileSystemDocumentLoader.loadDocument(path, parser);

            DocumentSplitter splitter = DocumentSplitters.recursive(600, 0);
            List<TextSegment> segments = splitter.split(document);

            Response<List<Embedding>> response = embeddingModel.embedAll(segments);
            List<Embedding> embeddings = response.content();

            EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
            embeddingStore.addAll(embeddings, segments);
            return embeddingStore;

        } catch (Exception e) {
            throw new RuntimeException("Erreur lors de l'ingestion de " + resourceName, e);
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