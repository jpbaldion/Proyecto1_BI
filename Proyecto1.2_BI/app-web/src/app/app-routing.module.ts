import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { InicioComponent } from './Inicio/Inicio.component';
import { PredictComponent } from './predict/predict.component';
import { PredictResultsComponent } from './predict-results/predict-results.component';

const routes: Routes = [
  { path: '', component: InicioComponent },
  { path: 'predict', component: PredictComponent },
  { path: 'predict-results', component: PredictResultsComponent}

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
