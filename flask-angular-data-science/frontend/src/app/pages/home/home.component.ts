import {Component, OnInit} from '@angular/core';
import {liver_dfService} from "./iris.service";
import {
    liver_df,
    ProbabilityPrediction,
    SVCParameters,
    SVCResult
} from "./types";

@Component({
    selector: 'home',
    templateUrl: './home.component.html',
    styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

    public svcParameters: SVCParameters = new SVCParameters();
    public svcResult: SVCResult;
    public iris: liver_df = new liver_df();
    public probabilityPredictions: ProbabilityPrediction[];

    // graph styling
    public colorScheme = {
        domain: ['#1a242c', '#e81746', '#e67303', '#f0f0f0']
    };

    constructor(private irisService: liver_dfService) {
    }

    ngOnInit() {
    }

    public trainModel() {
        this.irisService.trainModel(this.svcParameters).subscribe((svcResult) => {
            this.svcResult = svcResult;
        });
    }

    public predictliver_df() {
        this.irisService.predictIris(this.iris).subscribe((probabilityPredictions) => {
            this.probabilityPredictions = probabilityPredictions;
        });
    }

}
