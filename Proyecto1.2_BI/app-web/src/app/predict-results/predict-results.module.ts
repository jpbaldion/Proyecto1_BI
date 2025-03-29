import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictResultsComponent } from './predict-results.component';
import { NavbarModule } from '../Navbar/Navbar.module';

@NgModule({
  imports: [
    CommonModule,
    NavbarModule
  ],
  declarations: [PredictResultsComponent]
})
export class PredictResultsModule { }
