import express from "express";
import cors from "cors";
import * as faceapi from "face-api.js";
import path from "path";
import { fileURLToPath } from "url";
import canvas from "canvas";
import fetch from "node-fetch";

// Configure canvas for face-api
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData, fetch });

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Constants
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second
const FACE_MATCHER_THRESHOLD = 0.6;

const app = express();

// Middleware
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(cors());
app.use("/models", express.static(path.join(__dirname, "models")));

// Helper function to delay execution
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Enhanced image loading with retry logic
async function loadImageWithRetry(url, retries = MAX_RETRIES) {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      const response = await fetch(url, {
        timeout: 5000, // 5 second timeout
        headers: {
          "User-Agent": "Face-Recognition-API/1.0",
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const buffer = await response.buffer();
      const image = await canvas.loadImage(buffer);
      return image;
    } catch (error) {
      console.error(
        `Attempt ${attempt}/${retries} failed for URL ${url}:`,
        error.message
      );

      if (attempt === retries) {
        throw new Error(
          `Failed to load image after ${retries} attempts: ${error.message}`
        );
      }

      await delay(RETRY_DELAY * attempt); // Exponential backoff
    }
  }
}

// Validate dataset entry
function validateDatasetEntry(entry) {
  if (!entry || typeof entry !== "object") {
    throw new Error("Invalid dataset entry format");
  }

  if (!entry.id || !entry.imglink) {
    throw new Error("Missing required fields: id or imglink");
  }

  if (!/^https?:\/\/.+/.test(entry.imglink)) {
    throw new Error("Invalid image URL format");
  }

  return true;
}

// Process individual face with improved error handling
async function processSingleFace(data) {
  const { id, imglink } = data;

  try {
    validateDatasetEntry(data);

    console.log(`Processing image for id ${id}`);
    const img = await loadImageWithRetry(imglink);

    const detection = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();

    if (!detection) {
      throw new Error("No face detected in image");
    }

    return new faceapi.LabeledFaceDescriptors(id.toString(), [
      detection.descriptor,
    ]);
  } catch (error) {
    console.error(`Error processing image for id ${id}:`, error.message);
    return null;
  }
}

// Enhanced loadLabeledImages function
async function loadLabeledImages(dataset) {
  console.log(`Processing ${dataset.length} images from dataset`);

  const labeledDescriptors = await Promise.all(dataset.map(processSingleFace));

  const validDescriptors = labeledDescriptors.filter((desc) => desc !== null);
  console.log(
    `Generated ${validDescriptors.length} valid descriptors out of ${dataset.length} images`
  );

  if (validDescriptors.length === 0) {
    throw new Error(
      "No valid face descriptors could be generated from the dataset"
    );
  }

  return validDescriptors;
}

// Initialize face-api models
async function initializeModels() {
  const modelPath = path.join(__dirname, "models");

  try {
    await Promise.all([
      faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath),
      faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath),
      faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath),
    ]);
    console.log("Models loaded successfully");
  } catch (error) {
    throw new Error(`Failed to load face recognition models: ${error.message}`);
  }
}

// Main route handler
app.post("/", async (req, res) => {
  try {
    const { dataset, group_img } = req.body;

    // Input validation
    if (!dataset || !Array.isArray(dataset) || dataset.length === 0) {
      return res.status(400).json({
        error: "Invalid dataset format",
        details: "Dataset must be a non-empty array",
      });
    }

    if (!group_img || typeof group_img !== "string") {
      return res.status(400).json({
        error: "Invalid group_img",
        details: "group_img must be a valid URL string",
      });
    }

    // Initialize models
    await initializeModels();

    // Process individual faces
    const labeledFaceDescriptors = await loadLabeledImages(dataset);

    // Process group image
    const image = await loadImageWithRetry(group_img);
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();

    if (detections.length === 0) {
      return res.status(400).json({
        error: "No faces detected in group image",
      });
    }

    console.log(`Detected ${detections.length} faces in group image`);

    // Create face matcher with custom threshold
    const faceMatcher = new faceapi.FaceMatcher(
      labeledFaceDescriptors,
      FACE_MATCHER_THRESHOLD
    );

    // Match faces
    const results = detections.map((d) =>
      faceMatcher.findBestMatch(d.descriptor)
    );

    // Return results
    res.json({
      success: true,
      totalFacesDetected: detections.length,
      matches: results.map((result) => ({
        label: result.label,
        distance: result.distance,
        confidence: ((1 - result.distance) * 100).toFixed(2) + "%",
      })),
    });
  } catch (error) {
    console.error("Error in POST handler:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error.message,
      stack: process.env.NODE_ENV === "development" ? error.stack : undefined,
    });
  }
});

// Start server
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Models directory: ${path.join(__dirname, "models")}`);
});
