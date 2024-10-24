import express from "express";
import cors from "cors";

const app = express();

app.use(express.json());
app.use(cors());

app.post("/", (req, res) => {
  const { dataset, group_img } = req.body;

  Promise.all([
    faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
    faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
    faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
  ]).then(start);

  async function start(group_img_url) {
    // Assuming you pass the group_img_url when calling start()
    const labeledFaceDescriptors = await loadLabeledImages(); // Assumes dataset is passed in loadLabeledImages
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
    let image;
    // let canvas;
    // document.body.append("Loaded");

    // Fetch the group image from the provided URL
    try {
      image = await faceapi.fetchImage(group_img_url); // Use the group image URL from the backend
      //   container.append(image);
      //   canvas = faceapi.createCanvasFromMedia(image);
      //   container.append(canvas);

      //   const displaySize = { width: image.width, height: image.height };
      //   faceapi.matchDimensions(canvas, displaySize);

      // Detect all faces in the group image
      const detections = await faceapi
        .detectAllFaces(image)
        .withFaceLandmarks()
        .withFaceDescriptors();

      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      // Match the detected faces with the labeled descriptors
      const results = resizedDetections.map((d) =>
        faceMatcher.findBestMatch(d.descriptor)
      );

      return res.json(results);

      // Log recognized faces and draw boxes around them
      //   results.forEach((result, i) => {
      //     console.log(`Detected: ${result.label}`); // Logs the id from the dataset

      //     const box = resizedDetections[i].detection.box;
      //     const drawBox = new faceapi.draw.DrawBox(box, {
      //       label: result.toString(),
      //     });
      //     drawBox.draw(canvas);
      //   });
    } catch (error) {
      console.error("Error loading the group image:", error);
    }
  }

  function loadLabeledImages(dataset) {
    return Promise.all(
      dataset.map(async (data) => {
        const { id, imglink } = data; // Get the id and image link from the dataset
        const descriptions = [];

        // Assuming you want to fetch just one image per dataset entry
        const img = await faceapi.fetchImage(imglink);
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();

        if (detections) {
          descriptions.push(detections.descriptor);
        }

        // Return LabeledFaceDescriptors with the id as the label
        return new faceapi.LabeledFaceDescriptors(id, descriptions);
      })
    );
  }
});

app.listen(3000);
