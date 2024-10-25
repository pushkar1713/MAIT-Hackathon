// import express from "express";
// import cors from "cors";
// import * as faceapi from "face-api.js";
// import path from "path";
// import { fileURLToPath } from "url";

// // Get the __dirname equivalent in ES modules
// const __filename = fileURLToPath(import.meta.url);
// const __dirname = path.dirname(__filename);

// const app = express();

// // Middleware to parse JSON and enable CORS
// app.use(express.json());
// app.use(cors());
// console.log("Serving models from:", path.join(__dirname, "models"));

// // Serve models from the "models" directory using the ES module-compatible path
// app.use("/models", express.static(path.join(__dirname, "models")));

// app.post("/", (req, res) => {
//   const { dataset, group_img } = req.body;
//   console.log("Dataset received:", dataset);

//   // Load the models from the "models" directory
//   const serverUrl = "http://localhost:3000"; // Your server address

//   Promise.all([
//     faceapi.nets.faceRecognitionNet.loadFromUri(`${serverUrl}/models`),
//     faceapi.nets.faceLandmark68Net.loadFromUri(`${serverUrl}/models`),
//     faceapi.nets.ssdMobilenetv1.loadFromUri(`${serverUrl}/models`),
//   ])
//     .then(start)
//     .catch((error) => console.error("Error loading models:", error));

//   // Start face matching process
//   async function start(group_img_url, dataset) {
//     try {
//       const labeledFaceDescriptors = await loadLabeledImages(dataset); // Pass dataset
//       const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

//       // Fetch and process the group image
//       const image = await faceapi.fetchImage(group_img_url);
//       const displaySize = { width: image.width, height: image.height };

//       const detections = await faceapi
//         .detectAllFaces(image)
//         .withFaceLandmarks()
//         .withFaceDescriptors();

//       const resizedDetections = faceapi.resizeResults(detections, displaySize);

//       // Match faces with labeled face descriptors
//       const results = resizedDetections.map((d) =>
//         faceMatcher.findBestMatch(d.descriptor)
//       );

//       res.json(results); // Send the results back to the client
//     } catch (error) {
//       console.error("Error processing group image:", error);
//       res.status(500).send("Error processing group image.");
//     }
//   }

//   // Load labeled images (from the dataset) for face matching
//   function loadLabeledImages(dataset) {
//     return Promise.all(
//       dataset.map(async (data) => {
//         const { id, imglink } = data;
//         const descriptions = [];

//         try {
//           const img = await faceapi.fetchImage(imglink);
//           const detections = await faceapi
//             .detectSingleFace(img)
//             .withFaceLandmarks()
//             .withFaceDescriptor();

//           if (detections) {
//             descriptions.push(detections.descriptor);
//           }

//           // Return LabeledFaceDescriptors with the id as the label
//           return new faceapi.LabeledFaceDescriptors(
//             id.toString(),
//             descriptions
//           );
//         } catch (error) {
//           console.error(`Error processing image with id ${id}:`, error);
//           return new faceapi.LabeledFaceDescriptors(id.toString(), []); // Return empty descriptors on error
//         }
//       })
//     );
//   }
// });

// // Start the server on port 3000
// app.listen(3000, () => {
//   console.log("Server running on port 3000");
// });

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

const app = express();

// Middleware
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(cors());

// Serve models directory
app.use("/models", express.static(path.join(__dirname, "models")));

// Helper function to load image
async function loadImage(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.statusText}`);
    }
    const buffer = await response.buffer();
    const image = await canvas.loadImage(buffer);
    return image;
  } catch (error) {
    console.error("Error loading image:", error);
    throw error;
  }
}

// Helper function to get absolute model path
function getAbsoluteModelPath() {
  const modelPath = path.join(__dirname, "models");
  return path.join(__dirname, "models");
}

app.post("/", async (req, res) => {
  try {
    console.log("Received request");

    const { dataset, group_img } = req.body;

    // Validate input
    if (!dataset || !Array.isArray(dataset)) {
      return res.status(400).json({
        error: "Invalid dataset format",
        received: dataset,
      });
    }

    if (!group_img) {
      return res.status(400).json({
        error: "Missing group_img",
      });
    }

    // Load models using absolute paths
    console.log("Loading models...");
    const modelPath = getAbsoluteModelPath();
    console.log("Model path:", modelPath);

    try {
      await Promise.all([
        faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath),
        faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath),
        faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath),
      ]);
      console.log("Models loaded successfully");
    } catch (modelError) {
      console.error("Error loading models:", modelError);
      return res.status(500).json({
        error: "Failed to load face recognition models",
        message: modelError.message,
      });
    }

    // Process individual faces
    console.log("Processing individual faces...");
    const labeledFaceDescriptors = await loadLabeledImages(dataset);

    if (labeledFaceDescriptors.length === 0) {
      return res.status(400).json({
        error: "No valid face descriptors could be generated from the dataset",
      });
    }
    console.log(`Processed ${labeledFaceDescriptors.length} individual faces`);

    // Create face matcher
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

    // Process group image
    console.log("Processing group image...");
    try {
      const image = await loadImage(group_img);
      const detections = await faceapi
        .detectAllFaces(image)
        .withFaceLandmarks()
        .withFaceDescriptors();

      console.log(`Detected ${detections.length} faces in group image`);

      const results = detections.map((d) =>
        faceMatcher.findBestMatch(d.descriptor)
      );

      res.json({
        success: true,
        matches: results.map((result) => ({
          label: result.label,
          distance: result.distance,
        })),
      });
    } catch (groupImageError) {
      console.error("Error processing group image:", groupImageError);
      return res.status(500).json({
        error: "Failed to process group image",
        message: groupImageError.message,
      });
    }
  } catch (error) {
    console.error("Error in POST handler:", error);
    res.status(500).json({
      error: "Internal server error",
      message: error.message,
    });
  }
});

async function loadLabeledImages(dataset) {
  console.log(`Processing ${dataset.length} images from dataset`);

  const labeledDescriptors = await Promise.all(
    dataset.map(async (data) => {
      const { id, imglink } = data;

      if (!id || !imglink) {
        console.error("Invalid data entry:", data);
        return null;
      }

      try {
        console.log(`Processing image for id ${id}`);
        const img = await loadImage(imglink);

        const detection = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (!detection) {
          console.warn(`No face detected in image for id: ${id}`);
          return null;
        }

        return new faceapi.LabeledFaceDescriptors(id.toString(), [
          detection.descriptor,
        ]);
      } catch (error) {
        console.error(`Error processing image ${id}:`, error);
        return null;
      }
    })
  );

  // Filter out null results
  const validDescriptors = labeledDescriptors.filter((desc) => desc !== null);
  console.log(`Generated ${validDescriptors.length} valid descriptors`);

  return validDescriptors;
}

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Models directory: ${path.join(__dirname, "models")}`);
});
