import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { AppService } from '../app.service';

@Component({
  selector: 'app-predict-results',
  templateUrl: './predict-results.component.html',
  styleUrls: ['./predict-results.component.css']
})
export class PredictResultsComponent implements OnInit {
  responses: any[] = [];

  constructor(private router: Router, private appService: AppService) {}

  ngOnInit() {
    this.responses = this.appService.getPredictionData()
    if (this.responses.length === 0) {
      this.router.navigate(['/predict']);
    }
  }
}
