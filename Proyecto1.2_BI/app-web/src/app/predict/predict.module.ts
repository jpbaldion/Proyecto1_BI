import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { PredictComponent } from './predict.component';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';
import { NavbarModule } from '../Navbar/Navbar.module';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    HttpClientModule,
    NavbarModule
  ],
  declarations: [PredictComponent]
})
export class PredictModule { }
