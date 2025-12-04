import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Dual-output logger - writes to both console and log file
 * Used to create permanent records of experiment runs
 */
public class ExperimentLogger {
    private PrintStream originalOut;
    private PrintStream logFileStream;
    private Path logFile;
    
    /**
     * Start logging to file while still showing on console
     */
    public void startLogging(Path logFile) throws IOException {
        this.logFile = logFile;
        Files.createDirectories(logFile.getParent());
        
        // Save original System.out
        originalOut = System.out;
        
        // Create file output stream
        FileOutputStream fos = new FileOutputStream(logFile.toFile());
        logFileStream = new PrintStream(fos);
        
        // Create a PrintStream that writes to both console and file
        PrintStream dualStream = new PrintStream(new OutputStream() {
            @Override
            public void write(int b) throws IOException {
                originalOut.write(b);  // Write to console
                logFileStream.write(b); // Write to file
            }
            
            @Override
            public void flush() throws IOException {
                originalOut.flush();
                logFileStream.flush();
            }
        });
        
        System.setOut(dualStream);
        
        // Write header
        System.out.println("================================================================================");
        System.out.println("EXPERIMENT LOG");
        System.out.println("Started: " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        System.out.println("Log file: " + logFile);
        System.out.println("================================================================================");
        System.out.println();
    }
    
    /**
     * Stop logging and restore normal console output
     */
    public void stopLogging() {
        if (originalOut != null) {
            System.out.println();
            System.out.println("================================================================================");
            System.out.println("Completed: " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
            System.out.println("Log saved to: " + logFile);
            System.out.println("================================================================================");
            
            // Restore original System.out
            System.setOut(originalOut);
            
            // Close log file stream
            if (logFileStream != null) {
                logFileStream.close();
            }
        }
    }
    
    /**
     * Convenience method - log a single step
     */
    public static void logStep(Path logFile, Runnable task) throws IOException {
        ExperimentLogger logger = new ExperimentLogger();
        try {
            logger.startLogging(logFile);
            task.run();
        } catch (Exception e) {
            System.err.println("Error during logged execution: " + e.getMessage());
            e.printStackTrace();
            throw e;
        } finally {
            logger.stopLogging();
        }
    }
}
