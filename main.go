package main

import (
	"fmt"
	"os"

	"github.com/gocarina/gocsv"
)

type Iris struct {
	SepalLength float64 `csv:"sepal_length"`
	SepalWidth  float64 `csv:"sepal_width"`
	PetalLength float64 `csv:"petal_length"`
	PetalWidth  float64 `csv:"petal_width"`
	Species     string  `csv:"species"`
}

func main() {
	irisDataset := readCSV("assets/iris-dataset.csv")

	// specieses := []string{}
	// for _, iris := range irisDataset {
	// 	if !slices.Contains(specieses, iris.Species) {
	// 		specieses = append(specieses, iris.Species)
	// 	}
	// }

	epoch := 1000
	learningRate := .2
	weights := make([]float64, 15)

	fmt.Println(" [+] Start training the model")
	fmt.Println(" \tEpoch:", epoch)
	fmt.Println(" \tLearning rate:", learningRate)
	fmt.Println()

	fmt.Print("\tEpoch")
	for i := range weights {
		fmt.Print("\tw")
		fmt.Print(i + 1)
	}
	fmt.Println()

	for i := 0; i < epoch; i++ {
		for _, iris := range irisDataset {
			setosa := (iris.SepalLength * weights[0]) +
				(iris.SepalWidth * weights[3]) +
				(iris.PetalLength * weights[6]) +
				(iris.PetalWidth * weights[9]) +
				(1 * weights[12])
			versicolor := (iris.SepalLength * weights[1]) +
				(iris.SepalWidth * weights[4]) +
				(iris.PetalLength * weights[7]) +
				(iris.PetalWidth * weights[10]) +
				(1 * weights[13])
			virginica := (iris.SepalLength * weights[2]) +
				(iris.SepalWidth * weights[5]) +
				(iris.PetalLength * weights[8]) +
				(iris.PetalWidth * weights[11]) +
				(1 * weights[14])

			if setosa < 0 {
				setosa = 0
			} else {
				setosa = 1
			}

			if versicolor < 0 {
				versicolor = 0
			} else {
				versicolor = 1
			}

			if virginica < 0 {
				virginica = 0
			} else {
				virginica = 1
			}

			errorSetosa := 0.
			errorVersicolor := 0.
			errorVirginica := 0.

			if iris.Species == "setosa" {
				errorSetosa = 1 - setosa
				errorVersicolor = 0 - versicolor
				errorVirginica = 0 - virginica
			} else if iris.Species == "versicolor" {
				errorSetosa = 0 - setosa
				errorVersicolor = 1 - versicolor
				errorVirginica = 0 - virginica
			} else if iris.Species == "virginica" {
				errorSetosa = 0 - setosa
				errorVersicolor = 0 - versicolor
				errorVirginica = 1 - virginica
			}

			weights[0] = weights[0] + (learningRate * iris.SepalLength * errorSetosa)
			weights[1] = weights[1] + (learningRate * iris.SepalLength * errorVersicolor)
			weights[2] = weights[2] + (learningRate * iris.SepalLength * errorVirginica)
			weights[3] = weights[3] + (learningRate * iris.SepalWidth * errorSetosa)
			weights[4] = weights[4] + (learningRate * iris.SepalWidth * errorVersicolor)
			weights[5] = weights[5] + (learningRate * iris.SepalWidth * errorVirginica)
			weights[6] = weights[6] + (learningRate * iris.PetalLength * errorSetosa)
			weights[7] = weights[7] + (learningRate * iris.PetalLength * errorVersicolor)
			weights[8] = weights[8] + (learningRate * iris.PetalLength * errorVirginica)
			weights[9] = weights[9] + (learningRate * iris.PetalLength * errorSetosa)
			weights[10] = weights[10] + (learningRate * iris.PetalLength * errorVersicolor)
			weights[11] = weights[11] + (learningRate * iris.PetalLength * errorVirginica)
			weights[12] = weights[12] + (learningRate * 1 * errorSetosa)
			weights[13] = weights[13] + (learningRate * 1 * errorVersicolor)
			weights[14] = weights[14] + (learningRate * 1 * errorVirginica)
		}

		if (i+1)%50 == 0 {
			fmt.Print("\t")
			fmt.Print(i + 1)
			for _, weight := range weights {
				fmt.Print("\t")
				fmt.Printf("%.3f", weight)
			}
			fmt.Println()
		}
	}

	fmt.Println(" [+] Done training the model")
}

func readCSV(file string) []*Iris {
	irisDatasetFile, err := os.OpenFile(file, os.O_RDWR|os.O_CREATE, os.ModePerm)
	if err != nil {
		panic(err)
	}
	defer irisDatasetFile.Close()
	irisDataset := []*Iris{}

	if err := gocsv.UnmarshalFile(irisDatasetFile, &irisDataset); err != nil {
		panic(err)
	}

	return irisDataset
}
