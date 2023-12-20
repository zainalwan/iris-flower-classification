package main

import (
	"fmt"
	"os"
	"slices"

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

	specieses := []string{}
	for _, iris := range irisDataset {
		if !slices.Contains(specieses, iris.Species) {
			specieses = append(specieses, iris.Species)
		}
	}

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
			results := make([]float64, len(specieses))
			for j := range specieses {
				result := (iris.SepalLength * weights[0+j]) +
					(iris.SepalWidth * weights[3+j]) +
					(iris.PetalLength * weights[6+j]) +
					(iris.PetalWidth * weights[9+j]) +
					(1 * weights[12+j])
				if result < 0 {
					results[j] = 0
				} else {
					results[j] = 1
				}
			}

			errors := make([]float64, len(results))
			for j, species := range specieses {
				if iris.Species == species {
					errors[j] = 1 - results[j]
				} else {
					errors[j] = 0 - results[j]
				}
			}

			for j := range errors {
				weights[0+j] = weights[0+j] + (learningRate * iris.SepalLength * errors[j])
				weights[3+j] = weights[3+j] + (learningRate * iris.SepalWidth * errors[j])
				weights[6+j] = weights[6+j] + (learningRate * iris.PetalLength * errors[j])
				weights[9+j] = weights[9+j] + (learningRate * iris.PetalWidth * errors[j])
				weights[12+j] = weights[12+j] + (learningRate * 1 * errors[j])
			}
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

	fmt.Println(" [+] Start predict training data using trained model")
	correctPredict := 0
	for _, iris := range irisDataset {
		results := make([]float64, len(specieses))
		for j := range specieses {
			results[j] = (iris.SepalLength * weights[0+j]) +
				(iris.SepalWidth * weights[3+j]) +
				(iris.PetalLength * weights[6+j]) +
				(iris.PetalWidth * weights[9+j]) +
				(1 * weights[12+j])
		}

		result := ""
		if results[0] > results[1] && results[0] > results[2] {
			result = specieses[0]
		} else if results[1] > results[0] && results[1] > results[2] {
			result = specieses[1]
		} else if results[2] > results[0] && results[2] > results[1] {
			result = specieses[2]
		}

		if result == iris.Species {
			correctPredict++
		}
	}
	fmt.Println(" [+] Done predicting training data")
	fmt.Print(" \tAccuracy: ")
	fmt.Printf("%.2f", float64(correctPredict)/float64(len(irisDataset))*100)
	fmt.Println("%")
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
